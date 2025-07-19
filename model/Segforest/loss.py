import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleWeightedCELoss(nn.Module):
    def __init__(self, weights=[0.8, 0.13, 0.07]):
        super().__init__()
        self.weights = weights
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        # preds: list of [pred1, pred2, pred3], each (B, H, W) or (B, 1, H, W)
        # targets: list of [target1, target2, target3], each (B, H, W)
        total_loss = 0.0
        for i, (pred, t) in enumerate(zip(preds, targets)):
            # If target has shape (B, 1, H, W), squeeze to (B, H, W)
            if t.dim() == 4 and t.size(1) == 1:
                t = t.squeeze(1)
            # Resize target if needed (shouldn't be needed if targets are preprocessed, but keep for safety)
            if pred.shape[2:] != t.shape[1:]:
                t = F.interpolate(t.unsqueeze(1).float(), size=pred.shape[2:], mode='nearest').long().squeeze(1)
            loss = self.ce(pred, t)
            total_loss += self.weights[i] * loss
        return total_loss
