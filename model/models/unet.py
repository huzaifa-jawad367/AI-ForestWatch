import torch
import torch.nn as nn
from torchvision import models

from base import BaseModel
from .se_blocks import _make_se, MFFBlock, UNetSE_down_block, UNetSE_up_block


class UNet(BaseModel):
    """
    Unified UNet architecture supporting dynamic Multi-Feature Fusion (MFF)
    and Squeeze-and-Excitation (SE) blocks.

    Args:
        topology:       One of "ENC_1_DEC_1", "ENC_2_DEC_2", "ENC_3_DEC_3", "ENC_4_DEC_4".
        input_channels: Number of input bands / channels.
        num_classes:    Number of output segmentation classes.
        use_mff:        Boolean to enable MFF skip connections.
        use_se:         Boolean to enable SE blocks across the network.
        se_reduction:   SE squeeze ratio (default 16).
        se_flags:       Dict controlling where SE is applied (if use_se is True):
                            input      - SE on the raw input tensor
                            encoder    - SE inside each encoder block
                            decoder    - SE inside each decoder block
                            bottleneck - SE after the bottleneck convs
    """

    def __init__(self, topology, input_channels, num_classes,
                 use_mff=False, use_se=False, se_reduction=16, se_flags=None,
                 legacy_mff=False, input_mean=None, input_std=None, input_clip=10.0,
                 dropout_p=None):
        super(UNet, self).__init__()

        if (input_mean is None) != (input_std is None):
            raise ValueError("input_mean and input_std must be provided together")
        if input_mean is not None:
            if len(input_mean) != input_channels or len(input_std) != input_channels:
                raise ValueError("normalization statistics must match input_channels")
            mean = torch.as_tensor(input_mean, dtype=torch.float32).view(1, input_channels, 1, 1)
            std = torch.as_tensor(input_std, dtype=torch.float32).view(1, input_channels, 1, 1)
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("normalization statistics must be finite and std must be positive")
        else:
            mean = torch.empty(0, dtype=torch.float32)
            std = torch.empty(0, dtype=torch.float32)
        if input_clip is not None and input_clip <= 0:
            raise ValueError("input_clip must be positive or None")
        if dropout_p is not None and not 0 <= dropout_p < 1:
            raise ValueError("dropout_p must be in [0, 1) or None")
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)
        self.input_clip = input_clip

        self.use_mff = use_mff
        self.use_se = use_se

        # SE flag defaults
        if self.use_se and se_flags is None:
            self.se_flags = {
                "input": False,
                "encoder": True,
                "decoder": True,
                "bottleneck": False,
            }
        elif not self.use_se:
            self.se_flags = {
                "input": False,
                "encoder": False,
                "decoder": False,
                "bottleneck": False,
            }
        else:
            self.se_flags = se_flags

        # Topology dispatch table
        self.topologies = {
            "ENC_1_DEC_1": self.ENC_1_DEC_1,
            "ENC_2_DEC_2": self.ENC_2_DEC_2,
            "ENC_3_DEC_3": self.ENC_3_DEC_3,
            "ENC_4_DEC_4": self.ENC_4_DEC_4,
        }
        assert topology in self.topologies, \
            f"Unknown topology '{topology}'. Choose from {list(self.topologies)}"

        # ---- VGG-11 pretrained layers ----
        vgg_trained = models.vgg11(pretrained=True)
        pretrained_layers = list(vgg_trained.features)

        # ---- Common layers ----
        self.max_pool = nn.MaxPool2d(2, 2)
        # Keep historical defaults unless an experiment explicitly specifies
        # dropout, so old configs and checkpoints retain their original graph.
        effective_dropout_p = (
            float(dropout_p) if dropout_p is not None
            else (0.5 if (use_mff or use_se) else 0.6)
        )
        self.dropout = nn.Dropout2d(effective_dropout_p)
        self.activate = nn.ReLU(inplace=True)

        # ---- Optional SE on input ----
        self.se_input = (
            _make_se(input_channels, max(2, input_channels // 2))
            if self.se_flags.get("input", False) else None
        )

        # ---- Encoders (Dynamic SE) ----
        self.encoder_1 = UNetSE_down_block(
            input_channels, 64,
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", False),
        )
        self.encoder_2 = UNetSE_down_block(
            64, 128,
            conv_1=pretrained_layers[3],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", False),
        )
        self.encoder_3 = UNetSE_down_block(
            128, 256,
            conv_1=pretrained_layers[6],
            conv_2=pretrained_layers[8],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", False),
        )
        self.encoder_4 = UNetSE_down_block(
            256, 512,
            conv_1=pretrained_layers[11],
            conv_2=pretrained_layers[13],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", False),
        )

        # ---- Bottleneck convolutions ----
        self.mid_conv_64_64_a = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_64_64_b = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_128_128_a = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_128_128_b = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_256_256_a = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_256_256_b = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_512_1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv_1024_1024 = nn.Conv2d(1024, 1024, 3, padding=1)

        # Optional SE after bottleneck
        self.se_bottleneck = (
            _make_se(1024, se_reduction)
            if self.se_flags.get("bottleneck", False) else None
        )

        # ---- Optional MFF blocks ----
        if self.use_mff:
            enc_channels = [64, 128, 256, 512]
            if legacy_mff:
                self.mff1 = MFFBlock(enc_channels, 64)
                self.mff2 = MFFBlock(enc_channels, 128)
                self.mff3 = MFFBlock(enc_channels, 256)
                self.mff4 = MFFBlock(enc_channels, 512)
                prev_channels = {4: 512, 3: 256, 2: 128, 1: 64}
            else:
                mff_out = 64
                self.mff1 = MFFBlock(enc_channels, mff_out)
                self.mff2 = MFFBlock(enc_channels, mff_out)
                self.mff3 = MFFBlock(enc_channels, mff_out)
                self.mff4 = MFFBlock(enc_channels, mff_out)
                prev_channels = {4: mff_out, 3: mff_out, 2: mff_out, 1: mff_out}
        else:
            prev_channels = {4: 512, 3: 256, 2: 128, 1: 64}

        # ---- Decoders (Dynamic SE) ----
        self.decoder_4 = UNetSE_up_block(
            prev_channel=prev_channels[4],
            input_channel=self.mid_conv_1024_1024.out_channels,
            output_channel=256,
            se_reduction=se_reduction,
            use_se=self.se_flags.get("decoder", False),
        )
        self.decoder_3 = UNetSE_up_block(
            prev_channel=prev_channels[3],
            input_channel=self.decoder_4.output_channels,
            output_channel=128,
            se_reduction=se_reduction,
            use_se=self.se_flags.get("decoder", False),
        )
        self.decoder_2 = UNetSE_up_block(
            prev_channel=prev_channels[2],
            input_channel=self.decoder_3.output_channels,
            output_channel=64,
            se_reduction=se_reduction,
            use_se=self.se_flags.get("decoder", False),
        )
        self.decoder_1 = UNetSE_up_block(
            prev_channel=prev_channels[1],
            input_channel=self.decoder_2.output_channels,
            output_channel=64,
            se_reduction=se_reduction,
            use_se=self.se_flags.get("decoder", False),
        )

        # ---- Head ----
        self.binary_last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        # Set the forward method based on topology
        self.forward = self.topologies[topology]
        print('\n\n' + "#" * 100)
        print(f"(LOG): Unified UNet (MFF={self.use_mff}, SE={self.use_se})")
        print("(LOG): The following Model Topology will be Utilized: {}".format(
            self.forward.__name__))
        print("#" * 100 + '\n\n')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_input(self, x):
        if not self.input_mean.numel():
            return x
        x = (x - self.input_mean) / self.input_std
        if self.input_clip is not None:
            x = x.clamp(-self.input_clip, self.input_clip)
        return x

    def _maybe_se_input(self, x):
        return self.se_input(x) if self.se_input is not None else x

    def _maybe_se_bottleneck(self, x):
        return self.se_bottleneck(x) if self.se_bottleneck is not None else x

    # ------------------------------------------------------------------
    # Topology: 1 encoder → 1 decoder
    # ------------------------------------------------------------------
    def ENC_1_DEC_1(self, x_in):
        x_in = self._normalize_input(x_in)
        x_in = self._maybe_se_input(x_in)

        x1_cat = self.encoder_1(x_in)
        x1_cat_1 = self.dropout(x1_cat)
        x1 = self.max_pool(x1_cat_1)

        x_mid = self.activate(self.mid_conv_64_64_a(x1))
        x_mid = self.activate(self.mid_conv_64_64_b(x_mid))
        x_mid = self.dropout(x_mid)

        if self.use_mff:
            skip1 = self.mff1([x1_cat, x1_cat, x1_cat, x1_cat], target_size=x1_cat.shape[2:])
        else:
            skip1 = x1_cat

        x = self.decoder_1(skip1, x_mid)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

    # ------------------------------------------------------------------
    # Topology: 2 encoders → 2 decoders
    # ------------------------------------------------------------------
    def ENC_2_DEC_2(self, x_in):
        x_in = self._normalize_input(x_in)
        x_in = self._maybe_se_input(x_in)

        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)

        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)

        x_mid = self.activate(self.mid_conv_128_128_a(x2))
        x_mid = self.activate(self.mid_conv_128_128_b(x_mid))
        x_mid = self.dropout(x_mid)

        if self.use_mff:
            skip2 = self.mff2([x1_cat, x2_cat, x2_cat, x2_cat], target_size=x2_cat.shape[2:])
            skip1 = self.mff1([x1_cat, x2_cat, x2_cat, x2_cat], target_size=x1_cat.shape[2:])
        else:
            skip2 = x2_cat
            skip1 = x1_cat

        x = self.decoder_2(skip2, x_mid)
        x = self.decoder_1(skip1, x)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

    # ------------------------------------------------------------------
    # Topology: 3 encoders → 3 decoders
    # ------------------------------------------------------------------
    def ENC_3_DEC_3(self, x_in):
        x_in = self._normalize_input(x_in)
        x_in = self._maybe_se_input(x_in)

        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)

        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)

        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)

        x_mid = self.activate(self.mid_conv_256_256_a(x3))
        x_mid = self.activate(self.mid_conv_256_256_b(x_mid))
        x_mid = self.dropout(x_mid)

        if self.use_mff:
            skip3 = self.mff3([x1_cat, x2_cat, x3_cat, x3_cat], target_size=x3_cat.shape[2:])
            skip2 = self.mff2([x1_cat, x2_cat, x3_cat, x3_cat], target_size=x2_cat.shape[2:])
            skip1 = self.mff1([x1_cat, x2_cat, x3_cat, x3_cat], target_size=x1_cat.shape[2:])
        else:
            skip3 = x3_cat
            skip2 = x2_cat
            skip1 = x1_cat

        x = self.decoder_3(skip3, x_mid)
        x = self.decoder_2(skip2, x)
        x = self.decoder_1(skip1, x)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

    # ------------------------------------------------------------------
    # Topology: 4 encoders → 4 decoders  (full depth)
    # ------------------------------------------------------------------
    def ENC_4_DEC_4(self, x_in):
        x_in = self._normalize_input(x_in)
        x_in = self._maybe_se_input(x_in)

        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)

        x2_cat = self.encoder_2(x1)
        x2 = self.max_pool(self.dropout(x2_cat))

        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)

        x4_cat = self.encoder_4(x3)
        x4 = self.max_pool(self.dropout(x4_cat))

        x_mid = self.activate(self.mid_conv_512_1024(x4))
        x_mid = self.activate(self.mid_conv_1024_1024(x_mid))
        x_mid = self.dropout(x_mid)
        x_mid = self._maybe_se_bottleneck(x_mid)

        if self.use_mff:
            skip4 = self.mff4([x1_cat, x2_cat, x3_cat, x4_cat], target_size=x4_cat.shape[2:])
            skip3 = self.mff3([x1_cat, x2_cat, x3_cat, x4_cat], target_size=x3_cat.shape[2:])
            skip2 = self.mff2([x1_cat, x2_cat, x3_cat, x4_cat], target_size=x2_cat.shape[2:])
            skip1 = self.mff1([x1_cat, x2_cat, x3_cat, x4_cat], target_size=x1_cat.shape[2:])
        else:
            skip4 = x4_cat
            skip3 = x3_cat
            skip2 = x2_cat
            skip1 = x1_cat

        x = self.decoder_4(skip4, x_mid)
        x = self.decoder_3(skip3, x)
        x = self.decoder_2(skip2, x)
        x = self.decoder_1(skip1, x)

        x = self.binary_last_conv(x)
        return x, self.softmax(x)
