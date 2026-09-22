"""Dedicated modular UNet model with MFF skips and SE blocks enabled."""

from .unet import UNet


class UNetMFFSE(UNet):
    """UNet variant with multi-feature fusion and squeeze-excitation.

    Keeping this as a distinct model class makes the modularisation branch's
    factory explicit while preserving the parameter names and forward graph
    used by the historical UNetMFFSE checkpoints.
    """

    def __init__(self, topology, input_channels, num_classes,
                 se_reduction=16, se_flags=None, input_mean=None,
                 input_std=None, input_clip=10.0, dropout_p=None):
        super().__init__(
            topology=topology,
            input_channels=input_channels,
            num_classes=num_classes,
            use_mff=True,
            use_se=True,
            se_reduction=se_reduction,
            se_flags=se_flags,
            input_mean=input_mean,
            input_std=input_std,
            input_clip=input_clip,
            dropout_p=dropout_p,
        )
