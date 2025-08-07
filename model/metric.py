# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score, recall_score


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return precision_score(target.view(-1).cpu(), pred.view(-1).cpu())


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return recall_score(target.view(-1).cpu(), pred.view(-1).cpu())


def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1(target.view(-1).cpu(), pred.view(-1).cpu())


def accuracy(output, target):
    with torch.no_grad():
        # Handle both list and tensor inputs
        if isinstance(output, list):
            # If output is a list, use the first element
            pred = torch.argmax(output[0], dim=1)
        else:
            # If output is a tensor, use it directly
            pred = torch.argmax(output, dim=1)
        
        # print(f"type of pred: {type(pred)}")
        # print(f"shape of pred: {pred.shape}")
        # print(f"type of target: {type(target)}")
        # print(f"length of target: {len(target)}")
        # print(f"shape of target: {target.shape}")
        # print(f"type of output: {type(output)}")
        # print(f"length of output: {len(output)}")
        if isinstance(output, list):
            print(f"shape of output[0]: {output[0].shape}")
        else:
            print(f"shape of output: {output.shape}")
        
        assert pred.shape[0] == target.shape[0]  # Check batch size matches
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / (target.shape[1]*target.shape[2]*target.shape[0]) 
