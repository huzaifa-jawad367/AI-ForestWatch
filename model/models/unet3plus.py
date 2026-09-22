"""
Unified UNet3Plus architecture supporting dynamic Multi-Feature Fusion (MFF)
and Squeeze-and-Excitation (SE) blocks.
"""

import torch
import torch.nn as nn
from torchvision import models

from base import BaseModel
from .se_blocks import _make_se, MFFBlock, UNetSE_down_block, ConvBNReLU


class UNet3Plus(BaseModel):
    """
    Unified UNet 3+ architecture.
    Can toggle MFF and SE on/off.
    """

    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 use_mff: bool = False,
                 use_se: bool = False,
                 se_reduction: int = 16,
                 se_flags: dict = None,
                 dropout: float = 0.1):
        super(UNet3Plus, self).__init__()

        self.dropout_p = dropout
        self.se_reduction = se_reduction
        self.use_mff = use_mff
        self.use_se = use_se

        if self.use_se and se_flags is None:
            self.se_flags = {
                "input": False,
                "encoder": True,
                "fusion": False,
                "bottleneck": False
            }
        elif not self.use_se:
            self.se_flags = {
                "input": False,
                "encoder": False,
                "fusion": False,
                "bottleneck": False
            }
        else:
            self.se_flags = se_flags

        vgg_trained = models.vgg11(pretrained=True)
        pretrained_layers = list(vgg_trained.features)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(self.dropout_p)
        self.softmax = nn.Softmax(dim=1)
        self.activate = nn.ReLU(inplace=True)

        self.se_input = _make_se(
            input_channels,
            max(2, input_channels // 2)
        ) if self.se_flags.get("input", False) else None

        # ---------------- Encoder ----------------
        self.encoder_1 = UNetSE_down_block(
            input_channel=input_channels,
            output_channel=64,
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", False)
        )
        self.encoder_2 = UNetSE_down_block(
            input_channel=64,
            output_channel=128,
            conv_1=pretrained_layers[3],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", False)
        )
        self.encoder_3 = UNetSE_down_block(
            input_channel=128,
            output_channel=256,
            conv_1=pretrained_layers[6],
            conv_2=pretrained_layers[8],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", False)
        )
        self.encoder_4 = UNetSE_down_block(
            input_channel=256,
            output_channel=512,
            conv_1=pretrained_layers[11],
            conv_2=pretrained_layers[13],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", False)
        )

        self.bottom = ConvBNReLU(512, 1024, k=3, s=1, p=1,
                                 use_se=self.se_flags.get("bottleneck", False),
                                 se_reduction=self.se_reduction)

        self.se_bottleneck = _make_se(
            1024, se_reduction
        ) if self.se_flags.get("bottleneck", False) else None

        # ---------------- Aggregation / MFF ----------------
        agg_channels = 64
        # Channel counts from encoders: 64, 128, 256, 512, 1024 (bottom)

        if self.use_mff:
            # MFF Blocks replace separate projections and concatenation
            # One MFF block per decoder level
            # Decoder 4 inputs: e1, e2, e3, e4, bottom
            self.mff4 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff3 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff2 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff1 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
        else:
            # Standard independent projections
            self.proj1 = nn.Conv2d(64, agg_channels, kernel_size=1, bias=False)
            self.proj2 = nn.Conv2d(128, agg_channels, kernel_size=1, bias=False)
            self.proj3 = nn.Conv2d(256, agg_channels, kernel_size=1, bias=False)
            self.proj4 = nn.Conv2d(512, agg_channels, kernel_size=1, bias=False)
            self.proj5 = nn.Conv2d(1024, agg_channels, kernel_size=1, bias=False)

        # ---------------- Decoders ----------------
        # If MFF is used, input to decoder is the fused output (agg_channels)
        # If MFF is NOT used, input is concat of 5 projections (5 * agg_channels)
        dec_in_channels = agg_channels if self.use_mff else (agg_channels * 5)

        self.decoder_4 = ConvBNReLU(dec_in_channels, 512)
        self.decoder_3 = ConvBNReLU(dec_in_channels, 256)
        self.decoder_2 = ConvBNReLU(dec_in_channels, 128)
        self.decoder_1 = ConvBNReLU(dec_in_channels, 64)

        # SE on fusion (Only makes sense for standard concatenation, MFF has internal fusion)
        # We'll allow it if configured, but for MFF it applies to the MFF output
        self.se_fusion = _make_se(
            dec_in_channels, se_reduction
        ) if self.se_flags.get("fusion", False) else None

        if not self.use_mff:
            self.proj_d4 = nn.Conv2d(512, 64, kernel_size=1, bias=False)
            self.proj_d3 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
            self.proj_d2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        print('\n\n' + "#" * 100)
        print(f"(LOG): Unified UNet3Plus (MFF={self.use_mff}, SE={self.use_se})")
        print("#" * 100 + '\n\n')

    def _resize_to(self, x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _act_bn(self, bn_layer, x):
        return self.activate(bn_layer(x))

    def _maybe_se_input(self, x): return self.se_input(x) if self.se_input else x
    def _maybe_se_bottleneck(self, x): return self.se_bottleneck(x) if self.se_bottleneck else x
    def _maybe_se_fusion(self, x): return self.se_fusion(x) if self.se_fusion else x

    def forward(self, x_in):
        x_in = self._maybe_se_input(x_in)

        # Encoders
        e1 = self.encoder_1(x_in)
        e2 = self.encoder_2(self.max_pool(self.dropout(e1)))
        e3 = self.encoder_3(self.max_pool(self.dropout(e2)))
        e4 = self.encoder_4(self.max_pool(self.dropout(e3)))
        btm = self.bottom(self.max_pool(self.dropout(e4)))
        btm = self._maybe_se_bottleneck(btm)

        if self.use_mff:
            return self._forward_mff(e1, e2, e3, e4, btm)
        else:
            return self._forward_std(e1, e2, e3, e4, btm)

    def _forward_mff(self, e1, e2, e3, e4, btm):
        # D4
        # MFF inputs: [E1, E2, E3, E4, Btm]. Target size: E4 (1/8)
        f4 = self.mff4([e1, e2, e3, e4, btm], target_size=e4.shape[-2:])
        f4 = self._maybe_se_fusion(f4)
        d4 = self.decoder_4(f4)

        # D3
        # MFF inputs: [E1, E2, E3, D4, Btm]. Target size: E3 (1/4)
        f3 = self.mff3([e1, e2, e3, d4, btm], target_size=e3.shape[-2:])
        f3 = self._maybe_se_fusion(f3)
        d3 = self.decoder_3(f3)

        # D2
        # MFF inputs: [E1, E2, D3, E4, Btm] (replacing Slot 3 with D3?)
        f2 = self.mff2([e1, e2, d3, e4, btm], target_size=e2.shape[-2:])
        f2 = self._maybe_se_fusion(f2)
        d2 = self.decoder_2(f2)

        # D1
        # MFF inputs: [E1, D2, E3, E4, Btm] (replacing Slot 2 with D2?)
        f1 = self.mff1([e1, d2, e3, e4, btm], target_size=e1.shape[-2:])
        f1 = self._maybe_se_fusion(f1)
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
        f4 = self._maybe_se_fusion(f4)
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
        f3 = self._maybe_se_fusion(f3)
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
        f2 = self._maybe_se_fusion(f2)
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
        f1 = self._maybe_se_fusion(f1)
        d1 = self.decoder_1(f1)

        logits = self.head(d1)
        return logits, self.softmax(logits)
