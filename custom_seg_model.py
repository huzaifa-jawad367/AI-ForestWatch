import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from transformers import SegformerModel, SegformerDecodeHead
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Dict, Optional

def intersect_and_union(
    pred_label,
    label,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id

    if reduce_labels:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = label != ignore_index
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]

    area_intersect = torch.histc(intersect, bins=num_labels, min=0, max=num_labels-1)
    area_pred_label = torch.histc(pred_label, bins=num_labels, min=0, max=num_labels-1)
    area_label = torch.histc(label, bins=num_labels, min=0, max=num_labels-1)
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect.cpu(), area_union.cpu(), area_pred_label.cpu(), area_label.cpu()

def total_intersect_and_union(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    total_area_intersect = torch.zeros((num_labels,), dtype=torch.float64)
    total_area_union = torch.zeros((num_labels,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_labels,), dtype=torch.float64)
    total_area_label = torch.zeros((num_labels,), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            result, gt_seg_map, num_labels, ignore_index, label_map, reduce_labels
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label

def mean_iou(
    predictions,
    references,
    num_labels,
    ignore_index: bool,
    nan_to_num: Optional[int] = None,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = total_intersect_and_union(
        predictions, references, num_labels, ignore_index, label_map, reduce_labels
    )

    # compute metrics
    metrics = dict()

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label
    metrics["mean_iou"] = torch.nanmean(iou)
    metrics["mean_accuracy"] = torch.nanmean(acc)
    metrics["overall_accuracy"] = all_acc
    metrics["per_category_iou"] = iou
    metrics["per_category_accuracy"] = acc

    if nan_to_num is not None:
        metrics = dict(
            {metric: torch.nan_to_num(metric_value, nan=nan_to_num) for metric, metric_value in metrics.items()}
        )

    return metrics

def image_processor(image, label = None):
    if label is None:
        return {
            'pixel_values': image
        }
    return {
        'pixel_values': image,
        'labels': label
    }

class CustomSegModel(nn.Module):
    def __init__(self, config, optimizer=None, lr_scheduler=None, loss_fn = None, size = 128):
        super(CustomSegModel, self).__init__()
        self.config = config
        self.processor = image_processor 
        self.model = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.output_size = size

        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.33, verbose=True)

        self.scaler = torch.cuda.amp.GradScaler()
    
    def forward(self, images, labels=None, service='train'):
        if labels is not None:
            inputs = self.processor(images, labels)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            if service=='train':
                return self.train_forward(inputs)
            else:
                return self.eval_forward(inputs)
        else:
            inputs = self.processor(images)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            return self.predict(inputs)
    
    @torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False)
    def train_forward(self, inputs):
        outputs = self.model(inputs['pixel_values'].to(self.model.device),
                             output_attentions=False,
                             output_hidden_states=True,
                             return_dict=True)
        encoder_hidden_states = outputs.hidden_states
        logits = self.decode_head(encoder_hidden_states)

        loss = None
        labels = inputs['labels']
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
        )
        loss_fn = self.loss_fn
        loss = loss_fn(upsampled_logits, labels)
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=outputs.attentions,
        )
    
    @torch.no_grad()
    def eval_forward(self, inputs):
        labels = inputs['labels']
        outputs = self.model(inputs['pixel_values'], output_hidden_states=True, return_dict=True)
        logits = self.decode_head(outputs.hidden_states)
        return self.compute_metrics((logits, labels))
    
    @torch.no_grad()
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        # logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach()
        metrics = mean_iou(
            predictions=pred_labels,
            references=labels,
            num_labels=16,
            ignore_index=255,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics

    @torch.no_grad()
    def predict(self, inputs, target_sizes):
        outputs = self.model(inputs['pixel_values'], output_hidden_states=True, return_dict=True)
        logits = self.decode_head(outputs.hidden_states)
        outputs.logits = logits
        segmap = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        return segmap
    
    def train_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        
        self.scaler.scale(loss).backward()
        if batch_idx % self.config.accumulate == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        return outputs
    
    def train_one_epoch(self, loader, epoch):
        self.train()
        loss = 0
        total = 0
        with tqdm(loader) as t:
            t.set_description(f"Training {epoch}:")
            for i, batch in enumerate(t):
                outputs = self.train_step(batch, i + 1)
                # Avoid missing gradient
                self.optimizer.step()
                
                loss += outputs.loss.item()
                total += 1
                t.set_postfix(loss=f"{loss / total:.4f}")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(loss / total)
            self.on_epoch_end(epoch)
        return loss / total

    def eval_step(self, batch, batch_idx):
        return self(**batch, service='val')
    
    def eval_one_epoch(self, loader, epoch):
        self.eval()
        metrics = {'mean_iou': 0., 'mean_accuracy': 0., 'overall_accuracy': 0.}
        total = 0
        with tqdm(loader) as t:
            t.set_description(f"Validatring {epoch}:")
            for i, batch in enumerate(t):
                outputs = self.eval_step(batch, i)
                for k, v in outputs.items():
                    if k in metrics.keys():
                        metrics[k] += v
                total += 1
                t.set_postfix(
                            mean_iou=f"{metrics['mean_iou'] / total:.4f}",
                            mean_accuracy=f"{metrics['mean_accuracy'] / total:.4f}"
                )
#         self.writer.update(metrics, isval=True)

    def on_epoch_end(self, epoch):
        self.save(f"/kaggle/working/ckpt_{epoch}.pt")
        
    def save(self, filepath):
        self.config.from_zoo = None
        self.config.lr = self.optimizer.param_groups[0]['lr']
        torch.save({
            "config": self.config,
            "encoder_state_dict": self.model.state_dict(),
            "decoder_state_dict": self.decode_head.state_dict() 
        }, filepath)