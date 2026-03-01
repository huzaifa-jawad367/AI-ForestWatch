"""
UNet3PlusMFF — UNet 3+ variant with Multi-Feature Fusion (MFF), without Squeeze-and-Excitation.

- Encoder: UNet_down_block (VGG11-seeded, no SE)
- Bottleneck: Conv-BN-ReLU (no SE)
- Decoder: full-scale feature aggregation via MFF blocks
- Output: (logits, softmax(logits)) to match existing training loop
"""

import torch
import torch.nn as nn
from torchvision import models

from .unet import UNet_down_block
from .se_blocks import MFFBlock

try:
    from base.base_model import BaseModel
except ImportError:
    try:
        from base import BaseModel
    except ImportError:
        class BaseModel(nn.Module):
            def __init__(self):
                super(BaseModel, self).__init__()


class ConvBNReLU_NoSE(nn.Module):
    """Conv-BN-ReLU block without SE — lightweight version for non-SE models."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, stride=1, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3PlusMFF(BaseModel):
    """
    UNet 3+ with Multi-Feature Fusion (MFF) — no Squeeze-and-Excitation.

    Uses plain UNet_down_block encoders (VGG11-seeded) and MFFBlock for
    multi-scale feature aggregation in the decoder.
    """

    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_mff: bool = True):
        super(UNet3PlusMFF, self).__init__()

        self.dropout_p = dropout
        self.use_mff = use_mff

        vgg_trained = models.vgg11(pretrained=True)
        pretrained_layers = list(vgg_trained.features)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(self.dropout_p)
        self.softmax = nn.Softmax(dim=1)
        self.activate = nn.ReLU(inplace=True)

        # ---------------- Encoder (no SE) ----------------
        self.encoder_1 = UNet_down_block(input_channels, 64)
        self.encoder_2 = UNet_down_block(64, 128, conv_1=pretrained_layers[3])
        self.encoder_3 = UNet_down_block(
            128, 256, conv_1=pretrained_layers[6], conv_2=pretrained_layers[8])
        self.encoder_4 = UNet_down_block(
            256, 512, conv_1=pretrained_layers[11], conv_2=pretrained_layers[13])

        # Bottom (1/16) — no SE
        self.bottom = ConvBNReLU_NoSE(512, 1024, k=3, s=1, p=1)

        # ---------------- Aggregation / MFF ----------------
        agg_channels = 64

        if self.use_mff:
            # MFF Blocks: one per decoder level
            self.mff4 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff3 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff2 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff1 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
        else:
            # Standard independent projections (fallback)
            self.proj1 = nn.Conv2d(64, agg_channels, kernel_size=1, bias=False)
            self.proj2 = nn.Conv2d(128, agg_channels, kernel_size=1, bias=False)
            self.proj3 = nn.Conv2d(256, agg_channels, kernel_size=1, bias=False)
            self.proj4 = nn.Conv2d(512, agg_channels, kernel_size=1, bias=False)
            self.proj5 = nn.Conv2d(1024, agg_channels, kernel_size=1, bias=False)

        # ---------------- Decoders (no SE) ----------------
        dec_in_channels = agg_channels if self.use_mff else (agg_channels * 5)

        self.decoder_4 = ConvBNReLU_NoSE(dec_in_channels, 512)
        self.decoder_3 = ConvBNReLU_NoSE(dec_in_channels, 256)
        self.decoder_2 = ConvBNReLU_NoSE(dec_in_channels, 128)
        self.decoder_1 = ConvBNReLU_NoSE(dec_in_channels, 64)

        if not self.use_mff:
            self.proj_d4 = nn.Conv2d(512, 64, kernel_size=1, bias=False)
            self.proj_d3 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
            self.proj_d2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _resize_to(self, x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x_in):
        # Encoders
        e1 = self.encoder_1(x_in)
        e2 = self.encoder_2(self.max_pool(self.dropout(e1)))
        e3 = self.encoder_3(self.max_pool(self.dropout(e2)))
        e4 = self.encoder_4(self.max_pool(self.dropout(e3)))
        btm = self.bottom(self.max_pool(self.dropout(e4)))

        if self.use_mff:
            return self._forward_mff(e1, e2, e3, e4, btm)
        else:
            return self._forward_std(e1, e2, e3, e4, btm)

    def _forward_mff(self, e1, e2, e3, e4, btm):
        # D4
        f4 = self.mff4([e1, e2, e3, e4, btm], target_size=e4.shape[-2:])
        d4 = self.decoder_4(f4)

        # D3
        f3 = self.mff3([e1, e2, e3, d4, btm], target_size=e3.shape[-2:])
        d3 = self.decoder_3(f3)

        # D2
        f2 = self.mff2([e1, e2, d3, e4, btm], target_size=e2.shape[-2:])
        d2 = self.decoder_2(f2)

        # D1
        f1 = self.mff1([e1, d2, e3, e4, btm], target_size=e1.shape[-2:])
        d1 = self.decoder_1(f1)

        logits = self.head(d1)
        return logits, self.softmax(logits)

    def _forward_std(self, e1, e2, e3, e4, btm):
        # Projections
        p1 = self.proj1(e1)
        p2 = self.proj2(e2)
        p3 = self.proj3(e3)
        p4 = self.proj4(e4)
        p5 = self.proj5(btm)

        # D4
        t4 = e4
        f4 = torch.cat([
            self._resize_to(p1, t4),
            self._resize_to(p2, t4),
            self._resize_to(p3, t4),
            p4,
            self._resize_to(p5, t4),
        ], dim=1)
        d4 = self.decoder_4(f4)

        # D3
        t3 = e3
        d4_c = self._resize_to(self.proj_d4(d4), t3)
        f3 = torch.cat([
            self._resize_to(p1, t3),
            self._resize_to(p2, t3),
            p3,
            d4_c,
            self._resize_to(p5, t3),
        ], dim=1)
        d3 = self.decoder_3(f3)

        # D2
        t2 = e2
        d3_c = self._resize_to(self.proj_d3(d3), t2)
        f2 = torch.cat([
            self._resize_to(p1, t2),
            p2,
            d3_c,
            self._resize_to(p4, t2),
            self._resize_to(p5, t2),
        ], dim=1)
        d2 = self.decoder_2(f2)

        # D1
        t1 = e1
        d2_c = self._resize_to(self.proj_d2(d2), t1)
        f1 = torch.cat([
            p1,
            d2_c,
            self._resize_to(p3, t1),
            self._resize_to(p4, t1),
            self._resize_to(p5, t1),
        ], dim=1)
        d1 = self.decoder_1(f1)

        logits = self.head(d1)
        return logits, self.softmax(logits)


if __name__ == "__main__":

    print("\n" + "#" * 100)
    print("Testing UNet3PlusMFF architecture")
    print("#" * 100 + "\n")

    model = UNet3PlusMFF(input_channels=3, num_classes=2)
    print(model)

    x = torch.randn(1, 3, 128, 128)
    y, y_soft = model(x)
    print("\nOutput tensor shape (logits):", y.shape)
    print("Output tensor shape (softmax):", y_soft.shape)
