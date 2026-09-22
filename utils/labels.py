"""Label encoding shared by training, validation, metrics, and evaluation."""

from __future__ import annotations

import torch


DEFAULT_IGNORE_INDEX = -100


def encode_segmentation_target(
    target: torch.Tensor,
    *,
    mask_unknown_labels: bool,
    unknown_label: int = 0,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    num_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert source labels ``0, 1..C`` to training labels.

    Returns ``(encoded_target, labelled_mask)``. When masking is enabled,
    unknown pixels are assigned ``ignore_index``. When it is disabled, the
    historical mapping is retained for checkpoint/protocol compatibility.
    ``labelled_mask`` always identifies the source pixels with known labels.
    """

    if target.ndim == 4 and target.shape[1] == 1:
        source = target.squeeze(1)
    elif target.ndim == 3:
        source = target
    else:
        raise ValueError(
            "Expected segmentation target shaped [N, 1, H, W] or [N, H, W], "
            f"got {tuple(target.shape)}"
        )

    source = source.to(dtype=torch.long)
    labelled_mask = source != unknown_label

    if mask_unknown_labels:
        if labelled_mask.any():
            labelled = source[labelled_mask]
            if (labelled < 1).any() or (labelled > num_classes).any():
                values = torch.unique(labelled).detach().cpu().tolist()
                raise ValueError(
                    f"Known labels must be in [1, {num_classes}], got {values}"
                )
        encoded = torch.full_like(source, ignore_index)
        encoded[labelled_mask] = source[labelled_mask] - 1
    else:
        # Preserve the historical [0, 1, 2] -> [0, 0, 1] behavior unless a
        # versioned config explicitly opts into corrected masking.
        encoded = source.clone()
        encoded[encoded != unknown_label] -= 1

    return encoded, labelled_mask
