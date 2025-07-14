import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleWeightedCELoss(nn.Module):
    def __init__(self, weights=[0.8, 0.13, 0.07]):
        super().__init__()
        self.weights = weights
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, target):
        # outputs: list of [out1, out2, out3], each (B, C, H, W)
        # target: (B, H, W) or (B, 1, H, W)
        total_loss = 0.0
        for i, out in enumerate(outputs):
            # If target has shape (B, 1, H, W), squeeze to (B, H, W)
            t = target
            if t.dim() == 4 and t.size(1) == 1:
                t = t.squeeze(1)
            # Resize target if needed
            if out.shape[2:] != t.shape[1:]:
                t = F.interpolate(t.unsqueeze(1).float(), size=out.shape[2:], mode='nearest').long().squeeze(1)
            loss = self.ce(out, t)
            total_loss += self.weights[i] * loss
        return total_loss
