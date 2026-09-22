"""Explicit numeric-precision configuration for training and evaluation."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch


_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved precision behavior for one process."""

    mode: str
    autocast_dtype: torch.dtype
    use_autocast: bool
    use_grad_scaler: bool

    @property
    def display_name(self):
        return self.mode.upper()


def resolve_precision(
    trainer_config: Mapping,
    device: torch.device,
    *,
    bf16_supported: Optional[bool] = None,
) -> PrecisionPolicy:
    """Resolve FP32/FP16/BF16 without silently changing an explicit mode.

    ``trainer.precision`` is the preferred interface. The legacy
    ``trainer.amp`` flag remains supported: false means FP32, while true uses
    BF16 when the CUDA device supports it and FP16 otherwise. New controlled
    experiments should always set both fields consistently.
    """

    explicit = trainer_config.get("precision")
    legacy_amp = bool(trainer_config.get("amp", False))

    if explicit is None:
        if not legacy_amp:
            mode = "fp32"
        else:
            if device.type != "cuda":
                raise ValueError("trainer.amp=true requires a CUDA device")
            if bf16_supported is None:
                bf16_supported = torch.cuda.is_bf16_supported()
            mode = "bf16" if bf16_supported else "fp16"
    else:
        mode = str(explicit).strip().lower()
        if mode not in _DTYPES:
            raise ValueError(
                "trainer.precision must be one of: fp32, fp16, bf16; "
                f"got {explicit!r}"
            )
        expected_amp = mode != "fp32"
        if "amp" in trainer_config and legacy_amp != expected_amp:
            raise ValueError(
                f"Conflicting precision settings: precision={mode!r} "
                f"requires amp={str(expected_amp).lower()}"
            )

    if mode != "fp32" and device.type != "cuda":
        raise ValueError(f"trainer.precision={mode!r} requires a CUDA device")

    if mode == "bf16":
        if bf16_supported is None:
            bf16_supported = torch.cuda.is_bf16_supported()
        if not bf16_supported:
            raise RuntimeError(
                "BF16 was requested explicitly, but this CUDA device does not "
                "report BF16 support"
            )

    return PrecisionPolicy(
        mode=mode,
        autocast_dtype=_DTYPES[mode],
        use_autocast=mode != "fp32",
        # FP16 needs dynamic loss scaling. BF16 has FP32-like exponent range.
        use_grad_scaler=mode == "fp16",
    )


def create_grad_scaler(policy: PrecisionPolicy):
    """Create a CUDA GradScaler only when the resolved policy needs one."""

    if not policy.use_grad_scaler:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)
