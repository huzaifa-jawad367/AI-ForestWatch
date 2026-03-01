# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK)
# & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    UNet model definitions aggregated here with factory functions.
"""

import torch
import torch.nn as nn
from base import BaseModel
from torch.optim import *
from torchvision import models

# -----------------------------
# Import model variants
# -----------------------------
from .models.unet import UNet as UNetModel, UNet3Plus as UNet3PlusModel
from .models.unet_se_vgg import UNetSE as UNetSEModel
from .models.unet_se_resnet import UNetSE_resnet as UNetSEResnetModel
from .models.unet3plus_se import UNet3PlusSE as UNet3PlusSEModel
from .models.unet_mff import UNetMFF as UNetMFFModel
from .models.unet3plus_mff import UNet3PlusMFF as UNet3PlusMFFModel
from .models.segformer import CustomSegformer as CustomSegformerModel
from .models.unet3plus_mff_se import UNet3PlusMFFSE as UNet3PlusMFFSEModel


# -----------------------------
# Factory functions
# -----------------------------
def UNet(input_channels, num_classes, topology="ENC_4_DEC_4"):
    return UNetModel(topology=topology, input_channels=input_channels, num_classes=num_classes)


def UNet3Plus(input_channels, num_classes):
    return UNet3PlusModel(input_channels=input_channels, num_classes=num_classes)


def UNet3PlusMFF(input_channels, num_classes, use_mff=True):
    return UNet3PlusMFFModel(input_channels=input_channels, num_classes=num_classes, use_mff=use_mff)


def UNetSE(input_channels, num_classes, topology="ENC_4_DEC_4",
           se_reduction=16, se_flags=None):
    return UNetSEModel(topology=topology, input_channels=input_channels,
                       num_classes=num_classes, se_reduction=se_reduction,
                       se_flags=se_flags)


def UNet3PlusSE(input_channels, num_classes,
                se_reduction=16, se_flags=None):
    """
    Factory for UNet3PlusSE model (SE-enhanced UNet 3+)
    """
    return UNet3PlusSEModel(
        input_channels=input_channels,
        num_classes=num_classes,
        se_reduction=se_reduction,
        se_flags=se_flags
    )


def UNetMFF(input_channels, num_classes, topology="ENC_4_DEC_4"):
    return UNetMFFModel(topology=topology, input_channels=input_channels, num_classes=num_classes)


def UNetSEResnet(input_channels, num_classes, se_reduction=16, se_flags=None):
    return UNetSEResnetModel(input_channels=input_channels, num_classes=num_classes,
                             se_reduction=se_reduction, se_flags=se_flags)


def CustomSegformer(input_channels, num_classes, base_model='nvidia/mit-b0'):
    return CustomSegformerModel(input_channels=input_channels, num_classes=num_classes,
                                base_model=base_model)


def UNet3PlusMFFSE(input_channels, num_classes,
                   se_reduction=16, se_flags=None, use_mff=True):
    return UNet3PlusMFFSEModel(
        input_channels=input_channels,
        num_classes=num_classes,
        se_reduction=se_reduction,
        se_flags=se_flags,
        use_mff=use_mff
    )


# -----------------------------
# Model Checker
# -----------------------------
@torch.no_grad()
def check_model(model_type="UNet",
                topology="ENC_1_DEC_1",
                input_channels=7,
                num_classes=2,
                input_shape=[4, 7, 64, 64],
                base_model='nvidia/mit-b0',
                se_reduction=16,
                se_flags=None):

    if model_type == "UNet":
        model = UNet(topology=topology, input_channels=input_channels,
                     num_classes=num_classes)

    elif model_type == "UNet3Plus":
        model = UNet3Plus(input_channels=input_channels,
                          num_classes=num_classes)

    elif model_type == "UNet3PlusMFF":
        model = UNet3PlusMFF(input_channels=input_channels,
                             num_classes=num_classes)

    elif model_type == "UNetSE":
        model = UNetSE(topology=topology, input_channels=input_channels,
                       num_classes=num_classes,
                       se_reduction=se_reduction, se_flags=se_flags)

    elif model_type == "UNet3PlusSE":           # <-- ADDED HERE
        model = UNet3PlusSE(input_channels=input_channels,
                             num_classes=num_classes,
                             se_reduction=se_reduction,
                             se_flags=se_flags)

    elif model_type == "UNetMFF":
        model = UNetMFF(topology=topology, input_channels=input_channels,
                        num_classes=num_classes)

    elif model_type == "UNetSEResnet":
        model = UNetSEResnet(input_channels=input_channels,
                             num_classes=num_classes,
                             se_reduction=se_reduction,
                             se_flags=se_flags)

    elif model_type == "CustomSegformer":
        model = CustomSegformer(input_channels=input_channels,
                                num_classes=num_classes,
                                base_model=base_model)

    elif model_type == "UNet3PlusMFFSE":
        model = UNet3PlusMFFSE(input_channels=input_channels,
                               num_classes=num_classes,
                               se_reduction=se_reduction,
                               se_flags=se_flags)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Eval model and test forward pass
    model.eval()
    in_tensor = torch.Tensor(*input_shape)

    with torch.no_grad():
        out_tensor, softmaxed = model(in_tensor)
        print(in_tensor.shape, out_tensor.shape)


# -----------------------------
# Standalone testing
# -----------------------------
if __name__ == '__main__':

    INPUT_CHANNELS = 7
    NUM_CLASSES = 2
    INPUT_SHAPE = [4, INPUT_CHANNELS, 64, 64]

    def _count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def _test_model(name, model):
        print("\n" + "=" * 100)
        print(f"  MODEL: {name}")
        print("=" * 100)
        # print(model)  # Commented out: full model summary
        total, trainable = _count_params(model)
        print(f"\n  Total parameters:     {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Non-trainable params: {total - trainable:,}")

        model.eval()
        in_tensor = torch.randn(*INPUT_SHAPE)
        with torch.no_grad():
            out_tensor, softmaxed = model(in_tensor)
        print(f"  Input shape:  {list(in_tensor.shape)}")
        print(f"  Output shape: {list(out_tensor.shape)}")
        print("=" * 100 + "\n")

    # ---------- 1. UNet ----------
    _test_model("UNet", UNet(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        topology="ENC_4_DEC_4"
    ))

    # ---------- 1b. UNet3Plus ----------
    _test_model("UNet3Plus", UNet3Plus(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES
    ))

    # ---------- 1c. UNet3PlusMFF ----------
    _test_model("UNet3PlusMFF", UNet3PlusMFF(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES
    ))

    # ---------- 2. UNetSE (VGG backbone) ----------
    _test_model("UNetSE", UNetSE(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        topology="ENC_4_DEC_4"
    ))

    # ---------- 3. UNet3PlusSE ----------
    _test_model("UNet3PlusSE", UNet3PlusSE(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES
    ))

    # ---------- 4. UNetMFF ----------
    _test_model("UNetMFF", UNetMFF(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        topology="ENC_4_DEC_4"
    ))

    # ---------- 5. UNetSEResnet ----------
    try:
        _test_model("UNetSEResnet", UNetSEResnet(
            input_channels=INPUT_CHANNELS,
            num_classes=NUM_CLASSES
        ))
    except FileNotFoundError as e:
        print(f"\n  [SKIPPED] UNetSEResnet — MoCo checkpoint not found: {e}\n")

    # ---------- 6. CustomSegformer ----------
    try:
        _test_model("CustomSegformer", CustomSegformer(
            input_channels=INPUT_CHANNELS,
            num_classes=NUM_CLASSES,
            base_model='nvidia/mit-b3'
        ))
    except Exception as e:
        print(f"\n  [SKIPPED] CustomSegformer — {e}\n")
