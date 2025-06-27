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
            "pixel_values": image
        }
    else:
        return {
            "pixel_values": image,
            "labels": label
        }

class MySegFormer(nn.Module):
    def __init__(self, config, optimizer = None, lr_scheduler = None):
        super(MySegFormer, self).__init__()
        self.config = config
        self.encoder = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = 255)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)
        else:
            self.optimizer = optimizer
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = "min", factor = 0.5, patience = 5)

    def forward(self, input, service = "train"):
        input = {k: v.to(self.encoder.device) for k, v in input.items()}

        if service == "train":
            outputs = self.train_forward(input)
            return outputs
        else:
            outputs = self.eval_forward(input)
            return outputs
        
    @torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False)
    def train_forward(self, input):
        outputs = self.encoder(input["pixel_values"],
                               output_attentions = False,
                               output_hidden_states = True,
                               return_dict = True)
        
        logits = self.decode_head(outputs.hidden_states)

        labels = input["labels"]
        upsampled_logits = nn.functional.interpolate(logits, size = labels.shape[-2:], mode = "bilinear", align_corners = False)
        loss = self.loss_fn(upsampled_logits, labels)
        
        return SemanticSegmenterOutput(
            loss = loss,
            logits = logits,
            hidden_states = None,
            attentions=outputs.attentions)
    
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
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        outputs = self(batch, service = "train")
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_epoch(self, train_loader):
        self.train()
        losses = []
        eva_loss = None
        alpha = 0.9
        t = tqdm(train_loader, total = len(train_loader))
        for i, batch in enumerate(t):
            loss = self.train_step(batch)
            losses.append(loss)
            if eva_loss is None:
                eva_loss = loss
            else:
                eva_loss = alpha * eva_loss + (1 - alpha) * loss
            t.set_description(f"Train loss: {eva_loss:.5f}")
        return np.mean(losses)
    
    def eval_step(self, batch):
        self.eval()
        with torch.no_grad():
            outputs = self(batch, service = "eval")
            return outputs
        
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

    def on_epoch_end(self, epoch, val_loss = None):
        self.save(f"model/model_epoch_{epoch}")


    def save(self, filepath):
        torch.save({
            "config": self.config,
            "encoder_state_dict": self.model.state_dict(),
            "decoder_state_dict": self.decode_head.state_dict() 
        }, filepath)

    def load_pretrained(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.config = checkpoint["config"]
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decode_head.load_state_dict(checkpoint["decoder_state_dict"])

class Trainer:
    def __init__(self, model, train_loader, eval_loader, lr_scheduler = None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.lr_scheduler = lr_scheduler

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.model.train_epoch(self.train_loader)
            self.model.eval_one_epoch(self.eval_loader, epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(train_loss)
            self.model.on_epoch_end(epoch)
        self.model.save("model/model_final")

    def evaluate(self):
        self.model.eval_one_epoch(self.eval_loader, 0)

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, ignore_index = 255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        ce_loss = nn.functional.cross_entropy(logits, labels, ignore_index = self.ignore_index, reduction = "none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()
    

import torch

from dataset import SlidingWindowDataset
from transformers import SegformerConfig
from torch.utils.data import DataLoader
from torch.utils.data import random_split

CFG = SegformerConfig.from_pretrained("nvidia/mit-b5")
CFG.num_labels = 2
CFG.num_channels = 18

CFG.src_dir = "SRC"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_val_split = (0.9, 0.1)

dataset = SlidingWindowDataset(pickle_dir="data/train/training_2015_pickled_data", window_size=128, stride=64, reduce_indices=True)

generator = torch.Generator().manual_seed(123)
train_dataset, val_dataset = random_split(dataset, train_val_split, generator=generator)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = MySegFormer(CFG)

model.to(device)
model.load_pretrained("model/model_final")

trainer = Trainer(model, train_loader, val_loader)
trainer.evaluate()
