# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score, recall_score


IGNORE_INDEX = -100


def _labelled_vectors(output, target):
    pred = torch.argmax(output, dim=1).view(-1)
    target = target.view(-1)
    valid = target != IGNORE_INDEX
    return pred[valid], target[valid]


def precision(output, target):
    with torch.no_grad():
        pred, labelled = _labelled_vectors(output, target)
    if labelled.numel() == 0:
        return 0.0
    return precision_score(labelled.cpu(), pred.cpu(), zero_division=0.0)


def recall(output, target):
    with torch.no_grad():
        pred, labelled = _labelled_vectors(output, target)
    if labelled.numel() == 0:
        return 0.0
    return recall_score(labelled.cpu(), pred.cpu(), zero_division=0.0)


def f1_score(output, target):
    with torch.no_grad():
        pred, labelled = _labelled_vectors(output, target)
    if labelled.numel() == 0:
        return 0.0
    return f1(labelled.cpu(), pred.cpu(), zero_division=0.0)



def accuracy(output, target):
    with torch.no_grad():
        pred, labelled = _labelled_vectors(output, target)
        if labelled.numel() == 0:
            return 0.0
        correct = torch.sum(pred == labelled).item()
    return correct / labelled.numel()


def compute_iou(output, target):
    with torch.no_grad():
        num_classes = output.shape[1]
        pred_flat, target_flat = _labelled_vectors(output, target)

        iou_per_class = []
        for cls in range(num_classes):
            pred_mask = (pred_flat == cls)
            target_mask = (target_flat == cls)
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            if union == 0:
                # Skip classes not present in either pred or target
                continue
            iou_per_class.append(intersection / union)

    return sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0
