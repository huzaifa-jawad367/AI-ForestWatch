"""Global, unknown-masked metrics for binary semantic segmentation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


CLASS_NAMES = ("Non-Forest", "Forest")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _binary_erosion(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Erode ``[N,H,W]`` boolean masks with a square radius in pixels."""

    if radius <= 0:
        return mask
    inverse = (~mask).to(dtype=torch.float32).unsqueeze(1)
    inverse = F.pad(inverse, (radius, radius, radius, radius), value=1.0)
    dilated_inverse = F.max_pool2d(
        inverse, kernel_size=2 * radius + 1, stride=1
    )
    return dilated_inverse.squeeze(1) == 0


def _boundary_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return mask & ~_binary_erosion(mask, radius)


class MaskedSegmentationMetrics:
    """Accumulate one global confusion matrix and optional Boundary IoU counts.

    Targets must already use ``ignore_index`` for unknown/no-data pixels. A
    boundary width of 2% of the image diagonal follows the Boundary IoU paper's
    scale-relative convention. Pixels within that width of unknown regions or
    crop edges are excluded so invalid/crop boundaries are not scored as class
    boundaries.
    """

    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: int = -100,
        boundary_dilation_ratio: float = 0.02,
        compute_boundary: bool = True,
    ) -> None:
        if num_classes != 2:
            raise ValueError("This reporting layer currently expects two classes")
        if boundary_dilation_ratio <= 0:
            raise ValueError("boundary_dilation_ratio must be positive")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_dilation_ratio = boundary_dilation_ratio
        self.compute_boundary = compute_boundary
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.boundary_intersection = torch.zeros(num_classes, dtype=torch.int64)
        self.boundary_union = torch.zeros(num_classes, dtype=torch.int64)
        self.boundary_radius_pixels: set[int] = set()

    @torch.no_grad()
    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        prediction = output if output.ndim == 3 else torch.argmax(output, dim=1)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
            )
        prediction = prediction.to(dtype=torch.long)
        target = target.to(dtype=torch.long)
        valid = target != self.ignore_index

        if valid.any():
            encoded = target[valid] * self.num_classes + prediction[valid]
            counts = torch.bincount(
                encoded, minlength=self.num_classes * self.num_classes
            ).reshape(self.num_classes, self.num_classes)
            self.confusion += counts.detach().cpu()

        if not self.compute_boundary:
            return

        height, width = target.shape[-2:]
        radius = max(
            1,
            round(
                self.boundary_dilation_ratio
                * math.sqrt(height * height + width * width)
            ),
        )
        self.boundary_radius_pixels.add(radius)
        # Exclude a full boundary-width neighborhood around unknown pixels and
        # crop borders. Otherwise no-data holes/crop edges become fake class
        # boundaries and contaminate Boundary IoU.
        boundary_valid = _binary_erosion(valid, radius)

        for class_index in range(self.num_classes):
            target_mask = (target == class_index) & valid
            prediction_mask = (prediction == class_index) & valid
            target_boundary = _boundary_band(target_mask, radius) & boundary_valid
            prediction_boundary = (
                _boundary_band(prediction_mask, radius) & boundary_valid
            )
            intersection = (target_boundary & prediction_boundary).sum()
            union = (target_boundary | prediction_boundary).sum()
            self.boundary_intersection[class_index] += intersection.detach().cpu()
            self.boundary_union[class_index] += union.detach().cpu()

    def result(self) -> dict[str, float | int | list[int]]:
        confusion = self.confusion
        true_support = confusion.sum(dim=1)
        predicted_support = confusion.sum(dim=0)
        diagonal = confusion.diag()
        total = int(confusion.sum().item())

        precision = [
            _safe_ratio(int(diagonal[i]), int(predicted_support[i]))
            for i in range(self.num_classes)
        ]
        recall = [
            _safe_ratio(int(diagonal[i]), int(true_support[i]))
            for i in range(self.num_classes)
        ]
        f1 = [
            _safe_ratio(2 * precision[i] * recall[i], precision[i] + recall[i])
            if math.isfinite(precision[i]) and math.isfinite(recall[i])
            else float("nan")
            for i in range(self.num_classes)
        ]
        iou = [
            _safe_ratio(
                int(diagonal[i]),
                int(true_support[i] + predicted_support[i] - diagonal[i]),
            )
            for i in range(self.num_classes)
        ]
        dice = [
            _safe_ratio(
                2 * int(diagonal[i]), int(true_support[i] + predicted_support[i])
            )
            for i in range(self.num_classes)
        ]
        boundary_iou = [
            _safe_ratio(
                int(self.boundary_intersection[i]), int(self.boundary_union[i])
            )
            for i in range(self.num_classes)
        ]

        return {
            "accuracy": _safe_ratio(int(diagonal.sum()), total),
            "precision_non_forest": precision[0],
            "precision_forest": precision[1],
            "recall_non_forest": recall[0],
            "recall_forest": recall[1],
            "f1_non_forest": f1[0],
            "f1_forest": f1[1],
            "iou_non_forest": iou[0],
            "iou_forest": iou[1],
            "mean_iou": _nanmean(iou),
            "dice_non_forest": dice[0],
            "dice_forest": dice[1],
            "mean_dice": _nanmean(dice),
            "boundary_iou_non_forest": boundary_iou[0],
            "boundary_iou_forest": boundary_iou[1],
            "mean_boundary_iou": _nanmean(boundary_iou),
            "labelled_pixels": total,
            "boundary_radius_pixels": sorted(self.boundary_radius_pixels),
        }

    def legacy_names(self) -> dict[str, float]:
        """Map corrected global results onto existing log column names."""

        result = self.result()
        return {
            "accuracy": result["accuracy"],
            "f1_score": result["f1_forest"],
            "precision": result["precision_forest"],
            "recall": result["recall_forest"],
            "compute_iou": result["mean_iou"],
        }

    def classification_report(self) -> str:
        result = self.result()
        supports = self.confusion.sum(dim=1).tolist()
        total = sum(supports)
        weighted_precision = _safe_ratio(
            sum(
                (0.0 if not math.isfinite(result[f"precision_{key}"]) else result[f"precision_{key}"])
                * supports[index]
                for index, key in enumerate(("non_forest", "forest"))
            ),
            total,
        )
        weighted_recall = _safe_ratio(
            sum(
                (0.0 if not math.isfinite(result[f"recall_{key}"]) else result[f"recall_{key}"])
                * supports[index]
                for index, key in enumerate(("non_forest", "forest"))
            ),
            total,
        )
        weighted_f1 = _safe_ratio(
            sum(
                (0.0 if not math.isfinite(result[f"f1_{key}"]) else result[f"f1_{key}"])
                * supports[index]
                for index, key in enumerate(("non_forest", "forest"))
            ),
            total,
        )
        macro_precision = _nanmean(
            [result["precision_non_forest"], result["precision_forest"]]
        )
        macro_recall = _nanmean(
            [result["recall_non_forest"], result["recall_forest"]]
        )
        macro_f1 = _nanmean([result["f1_non_forest"], result["f1_forest"]])
        return (
            "\n"
            "              precision    recall  f1-score   support\n\n"
            f"  Non-Forest       {result['precision_non_forest']:.4f}    {result['recall_non_forest']:.4f}    {result['f1_non_forest']:.4f}  {supports[0]:>9d}\n"
            f"      Forest       {result['precision_forest']:.4f}    {result['recall_forest']:.4f}    {result['f1_forest']:.4f}  {supports[1]:>9d}\n\n"
            f"    accuracy                           {result['accuracy']:#.4g}  {total:>9d}\n"
            f"   macro avg       {macro_precision:.4f}    {macro_recall:.4f}    {macro_f1:.4f}  {total:>9d}\n"
            f"weighted avg       {weighted_precision:.4f}    {weighted_recall:.4f}    {weighted_f1:.4f}  {total:>9d}"
        )
