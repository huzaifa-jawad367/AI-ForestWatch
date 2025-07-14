import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math


class MultiScaleMultiDecoder(nn.Module):
    def __init__(self, mff_channels, tb4_channels, inner_channels=64, num_classes=3):
        """
        mff_channels: list of channels for [MFF_1, MFF_2, MFF_3]
        tb4_channels: number of channels in TB4
        inner_channels: number of channels after intermediate convolutions
        num_classes: number of output classes
        """
        super().__init__()
        # For TB4 upsampling
        self.tb4_up = nn.ConvTranspose2d(tb4_channels, tb4_channels, kernel_size=2, stride=2)
        # For MFF_3 concat TB4
        self.mff3_tb4_conv3x3 = nn.Conv2d(mff_channels[2] + tb4_channels, inner_channels, kernel_size=3, padding=1)
        self.mff3_tb4_up = nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=2, stride=2)
        self.mff3_tb4_out3_conv3x3 = nn.Conv2d(mff_channels[2] + tb4_channels, inner_channels, kernel_size=3, padding=1)
        self.mff3_tb4_out3_conv1x1 = nn.Conv2d(inner_channels, num_classes, kernel_size=1)
        # For MFF_2 concat x
        self.mff2_x_conv3x3 = nn.Conv2d(mff_channels[1] + inner_channels, inner_channels, kernel_size=3, padding=1)
        self.mff2_x_up = nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=2, stride=2)
        self.mff2_x_out2_conv3x3 = nn.Conv2d(mff_channels[2] + tb4_channels, inner_channels, kernel_size=3, padding=1)
        self.mff2_x_out2_conv1x1 = nn.Conv2d(inner_channels, num_classes, kernel_size=1)
        # For MFF_1 concat x
        self.mff1_x_conv3x3 = nn.Conv2d(mff_channels[0] + inner_channels, inner_channels, kernel_size=3, padding=1)
        self.mff1_x_out1_conv1x1 = nn.Conv2d(inner_channels, num_classes, kernel_size=1)

    def forward(self, mff_outputs, encoder_outputs):
        # mff_outputs: [MFF_1, MFF_2, MFF_3]
        # encoder_outputs: [TB1, TB2, TB3, TB4]
        TB4 = encoder_outputs[3]
        MFF_1, MFF_2, MFF_3 = mff_outputs
        # Step 1: x and out3
        tb4_up = self.tb4_up(TB4)
        mff3_tb4_cat = torch.cat([MFF_3, tb4_up], dim=1)
        x = self.mff3_tb4_conv3x3(mff3_tb4_cat)
        x = F.relu(x, inplace=True)
        x = self.mff3_tb4_up(x)
        out3 = self.mff3_tb4_out3_conv3x3(mff3_tb4_cat)
        out3 = F.relu(out3, inplace=True)
        out3 = self.mff3_tb4_out3_conv1x1(out3)
        # Step 2: x and out2
        mff2_x_cat = torch.cat([MFF_2, x], dim=1)
        x2 = self.mff2_x_conv3x3(mff2_x_cat)
        x2 = F.relu(x2, inplace=True)
        x2 = self.mff2_x_up(x2)
        out2 = self.mff2_x_out2_conv3x3(mff3_tb4_cat)
        out2 = F.relu(out2, inplace=True)
        out2 = self.mff2_x_out2_conv1x1(out2)
        # Step 3: out1
        mff1_x_cat = torch.cat([MFF_1, x2], dim=1)
        out1 = self.mff1_x_conv3x3(mff1_x_cat)
        out1 = F.relu(out1, inplace=True)
        out1 = self.mff1_x_out1_conv1x1(out1)
        if self.training:
            return [out1, out2, out3]
        else:
            return [out1]

