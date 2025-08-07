import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from mmcv.runner import load_checkpoint
import math

class MixTransformerFeatureConcat(nn.Module):
    """
    Takes the outputs of MixVisionTransformer (a list of 4 tensors) and an integer k (1-3),
    resizes all outputs to the spatial size of outs[k-1], and concatenates them along the channel dimension.
    """
    def __init__(self):
        super().__init__()

    def forward(self, outs, k):
        # outs: list of 4 tensors, each of shape (B, C_i, H_i, W_i)
        # k: int in 1, 2, 3 (corresponds to index 0, 1, 2)
        assert isinstance(outs, (list, tuple)) and len(outs) == 4, "outs must be a list of 4 tensors"
        assert 1 <= k <= 3, "k must be in 1, 2, 3"
        target_size = outs[k-1].shape[2:]
        resized = [F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False) for feat in outs]
        concat = torch.cat(resized, dim=1)
        return concat

class MFFBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv1x1(x)
        x = F.relu(x, inplace=True)
        x = self.conv3x3(x)
        return x

class MFFBlocks(nn.Module):
    """
    Takes the outputs of MixVisionTransformer, applies MixTransformerFeatureConcat for k=1,2,3,
    then passes each concatenated result through individual MFF blocks, returning a list of 3 outputs.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.feature_concat = MixTransformerFeatureConcat()
        # Create 3 MFF blocks, one for each k value
        self.mff_blocks = nn.ModuleList([
            MFFBlock(in_channels_list[i], out_channels[i]) for i in range(3)
        ])

    def forward(self, outs):
        # outs: list of 4 tensors from MixVisionTransformer
        # print("--------------------------------")
        # print("MFFBlocks forward")
        # print("--------------------------------")
        mff_outputs = []
        for k in range(1, 4):  # k = 1, 2, 3
            # Concatenate features for this k value
            concatenated = self.feature_concat(outs, k)
            # Pass through the corresponding MFF block
            mff_output = self.mff_blocks[k-1](concatenated)
            # print(f"Shape of MFF block output {k}: {mff_output.shape}")
            mff_outputs.append(mff_output)
        return mff_outputs

