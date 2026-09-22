"""
Standalone evaluation script for AI-ForestWatch models.

Loads a saved checkpoint, evaluates on the test split, and prints
all metrics plus a full classification report.

Usage:
    python evaluate.py -c saved/models/<run>/config.json \
                       -m saved/models/<run>/model_best.pth

    # Or use defaults for the Segformer run:
    python evaluate.py
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.segmentation_metrics import MaskedSegmentationMetrics
from utils import encode_segmentation_target, read_json


# ── reproducibility ─────────────────────────────────────────────────
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
np.random.seed(SEED)


def loss_input_for_model(model, logits, probabilities):
    return logits


def build_model(config_dict, checkpoint_path, device):
    """Instantiate the model from config and load checkpoint weights."""
    arch_cfg = config_dict["arch"]
    model = getattr(module_arch, arch_cfg["type"])(**arch_cfg["args"])

    from utils.training_protocol import load_checkpoint
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint.get("epoch", "?")
        print(f"  Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("  Loaded raw state_dict (no epoch info)")

    return model.to(device)


def build_test_loader(config_dict):
    """Build the test data loader from config."""
    dl_cfg = config_dict["train_data_loader"]
    loader = getattr(module_data, dl_cfg["type"])(**dl_cfg["args"], mode="test")
    return loader


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    criterion,
    metric_fns,
    device,
    use_amp=False,
    mask_unknown_labels=True,
    unknown_label=0,
    ignore_index=-100,
    boundary_dilation_ratio=0.02,
    artifact_writer=None,
):
    """Evaluate globally on labelled pixels only.

    ``mask_unknown_labels`` remains in the signature for call compatibility,
    but evaluation intentionally ignores its value: unknown labels are never a
    valid evaluation class.
    """
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    model.eval()

    # accumulators
    running_loss = 0.0
    aggregation_weight = 0
    global_metrics = MaskedSegmentationMetrics(
        ignore_index=ignore_index,
        boundary_dilation_ratio=boundary_dilation_ratio,
        compute_boundary=True,
    )

    pbar = tqdm(data_loader, desc="Evaluating", ncols=100, unit="batch", dynamic_ncols=True)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out_x, softmaxed = model(data)
            loss_target, label_valid = encode_segmentation_target(
                target,
                mask_unknown_labels=True,
                unknown_label=unknown_label,
                ignore_index=ignore_index,
                num_classes=softmaxed.shape[1],
            )
            loss = criterion(loss_input_for_model(model, out_x, softmaxed), loss_target)

        supervised_count = int(label_valid.sum().item())
        if supervised_count > 0:
            running_loss += loss.item() * supervised_count
            aggregation_weight += supervised_count
        global_metrics.update(softmaxed, loss_target)
        if artifact_writer is not None:
            artifact_writer.consume(data, target, softmaxed, evaluation_metrics=global_metrics)

        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

    pbar.close()

    # aggregate
    if aggregation_weight == 0:
        raise ValueError("Evaluation split contains no labelled pixels")
    avg_loss = running_loss / aggregation_weight
    result = global_metrics.result()
    avg_metrics = global_metrics.legacy_names()
    cls_report = global_metrics.classification_report()
    conf_mat = global_metrics.confusion.numpy()
    additional_metrics = {
        "iou_per_class": {
            "Non-Forest": result["iou_non_forest"],
            "Forest": result["iou_forest"],
        },
        "dice_per_class": {
            "Non-Forest": result["dice_non_forest"],
            "Forest": result["dice_forest"],
        },
        "boundary_iou_per_class": {
            "Non-Forest": result["boundary_iou_non_forest"],
            "Forest": result["boundary_iou_forest"],
        },
        "mean_iou": result["mean_iou"],
        "mean_dice": result["mean_dice"],
        "mean_boundary_iou": result["mean_boundary_iou"],
        "labelled_pixels": result["labelled_pixels"],
        "boundary_radius_pixels": result["boundary_radius_pixels"],
    }

    return avg_loss, avg_metrics, cls_report, conf_mat, additional_metrics, None, None



def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    parser.add_argument(
        "-c", "--config",
        default="saved/models/Landsat8_CustomSegformer_MS/CustomSegformer/0628_210105/config.json",
        type=str, help="Path to the run config.json",
    )
    parser.add_argument(
        "-m", "--model",
        default="saved/models/Landsat8_CustomSegformer_MS/CustomSegformer/0628_210105/model_best.pth",
        type=str, help="Path to the model checkpoint (.pth)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False,
        help="Enable automatic mixed precision during evaluation",
    )
    args = parser.parse_args()

    # ── load config ─────────────────────────────────────────────────
    config_dict = read_json(Path(args.config))
    if config_dict.get('protocol_id') is not None:
        from utils.training_protocol import validate_training_config
        validate_training_config(config_dict)
    else:
        print('Historical checkpoint: evaluation loss now uses logits; old probability-input losses are not comparable.')
    print(f"\n{'='*60}")
    print(f"  Evaluation Configuration")
    print(f"{'='*60}")
    print(f"  Config    : {args.config}")
    print(f"  Checkpoint: {args.model}")
    print(f"  Model     : {config_dict['arch']['type']}")
    print(f"  AMP       : {args.amp}")

    # ── device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device    : {device}")

    # ── model ───────────────────────────────────────────────────────
    print(f"\n  Loading model …")
    model = build_model(config_dict, args.model, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── data ────────────────────────────────────────────────────────
    print(f"\n  Building test data loader …")
    test_loader = build_test_loader(config_dict)
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Test batches: {len(test_loader)}")

    # ── loss & metrics ──────────────────────────────────────────────
    criterion = getattr(module_loss, config_dict["loss"])
    metric_fns = [getattr(module_metric, m) for m in config_dict["metrics"]]

    # ── evaluate ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Running evaluation on test set …")
    print(f"{'='*60}\n")

    use_amp = args.amp or config_dict.get("trainer", {}).get("amp", False)
    trainer_config = config_dict.get("trainer", {})
    artifact_writer = None
    if config_dict.get('visualizations', {}).get('enabled', True):
        from visualisation.paper_artifacts import PaperArtifacts
        artifact_writer = PaperArtifacts(model, test_loader, config_dict, args.model)
    avg_loss, avg_metrics, cls_report, conf_mat, additional_metrics, _, _ = evaluate(
        model,
        test_loader,
        criterion,
        metric_fns,
        device,
        use_amp=use_amp,
        mask_unknown_labels=True,
        unknown_label=trainer_config.get("unknown_label", 0),
        ignore_index=trainer_config.get("ignore_index", -100),
        boundary_dilation_ratio=trainer_config.get("boundary_dilation_ratio", 0.02),
        artifact_writer=artifact_writer,
    )

    # ── report ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Loss       : {avg_loss:.6f}")
    for name, val in avg_metrics.items():
        print(f"  {name:13s}: {val:.6f}")
    print(f"  Mean IoU   : {additional_metrics['mean_iou']:.6f}")
    print(f"  Mean Dice  : {additional_metrics['mean_dice']:.6f}")
    print(f"  Mean Boundary IoU: {additional_metrics['mean_boundary_iou']:.6f}")
    print(f"  Labelled pixels  : {additional_metrics['labelled_pixels']}")
    print(f"  Boundary radius  : {additional_metrics['boundary_radius_pixels']} pixels")
    for cls_name in ["Non-Forest", "Forest"]:
        print(f"  IoU ({cls_name:<10}): {additional_metrics['iou_per_class'][cls_name]:.6f}")
        print(f"  Dice ({cls_name:<9}): {additional_metrics['dice_per_class'][cls_name]:.6f}")
        print(
            f"  Boundary IoU ({cls_name:<10}): "
            f"{additional_metrics['boundary_iou_per_class'][cls_name]:.6f}"
        )

    print(f"\n  Classification Report:")
    print(cls_report)

    print(f"  Confusion Matrix:")
    print(f"  (rows = actual, cols = predicted)")
    print(f"               Non-Forest   Forest")
    print(f"  Non-Forest   {conf_mat[0][0]:>10d}   {conf_mat[0][1]:>6d}")
    print(f"  Forest       {conf_mat[1][0]:>10d}   {conf_mat[1][1]:>6d}")
    print(f"{'='*60}\n")

    if artifact_writer is not None:
        output = artifact_writer.finish({'loss': avg_loss, **avg_metrics,
            **additional_metrics, 'classification_report': cls_report,
            'confusion_matrix': conf_mat.tolist()})
        print(f'Paper visualizations: {output}')



if __name__ == "__main__":
    main()
