import os
from typing import Sequence, Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchgeo.datasets import LoveDA


class LoveDASegmentationDataset(Dataset):
    """
    A thin wrapper around torchgeo.datasets.LoveDA to expose it as a
    plain torch.utils.data.Dataset that returns (image, mask) tuples.
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        scene: Sequence[str] = ("urban", "rural"),
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """
        Args:
            root: Path where the LoveDA data will be stored/downloaded.
            split: One of "train", "val", or "test".
            scene: Which scenes to include: 'urban', 'rural', or both.
            transforms: A callable that takes and returns a dict with 'image' and 'mask'.
            download: If True, downloads the dataset if not already present.
            checksum: If True, verifies MD5 hashes on download.
        """
        super().__init__()
        self._base = LoveDA(
            root=root,
            split=split,
            scene=list(scene),
            transforms=transforms,
            download=download,
            checksum=checksum,
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        sample = self._base[idx]
        image = sample["image"]
        mask = sample.get("mask", None)
        return image, mask
