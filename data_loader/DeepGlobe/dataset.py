import torch
from torch.utils.data import Dataset
from torchgeo.datasets import DeepGlobeLandCover
from torch import Tensor
from typing import Callable, Optional


class DeepGlobeForestDataset(Dataset):
    """
    Filters DeepGlobeLandCover dataset to include only samples containing 'Forest land'
    and converts masks to binary: forest=1, everything else=0.
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        self._base = DeepGlobeLandCover(
            root=root,
            split=split,
            transforms=transforms,
            checksum=checksum,
        )

        self.forest_class = 3  # From DeepGlobeLandCover.classes

        # Pre-filter indices for efficiency
        self.indices = []
        for i in range(len(self._base)):
            sample = self._base[i]
            mask = sample["mask"]
            if (mask == self.forest_class).any():
                self.indices.append(i)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        actual_idx = self.indices[idx]
        sample = self._base[actual_idx]
        image, mask = sample["image"], sample["mask"]

        binary_mask = (mask == self.forest_class).long()  # 1 for forest, 0 otherwise

        return image, binary_mask


if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    try:
        ds = DeepGlobeForestDataset(root="data", split="train", transforms=None)
        print(f"[OK] Dataset instantiated. Forest-only samples: {len(ds)}")
        assert len(ds) > 0, "No samples with forest found. Is the dataset extracted under ./data/data/?"

        # --- Single sample check ---
        img, m = ds[0]
        print(f"Single sample -> image dtype/shape: {img.dtype}/{tuple(img.shape)} "
              f"| mask dtype/shape: {m.dtype}/{tuple(m.shape)}")

        # mask is binary 0/1
        uniques = torch.unique(m).tolist()
        print(f"Mask unique values: {uniques}")
        assert set(uniques).issubset({0, 1}), "Mask should be binary (0/1)!"

        # --- DataLoader smoke test ---
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
        imgs, masks = next(iter(loader))
        print(f"Batch -> images: {tuple(imgs.shape)}, masks: {tuple(masks.shape)}")

        # quick forest pixel ratio (0..1) in first batch
        forest_ratio = masks.float().mean().item()
        print(f"Forest pixel ratio (batch 0): {forest_ratio:.4f}")

        # --- Optional: visualize first sample (won't crash if headless) ---
        try:
            import matplotlib.pyplot as plt

            vis_img = torch.clamp(imgs[0], 0, 255).permute(1, 2, 0).byte().cpu().numpy()
            vis_mask = masks[0].cpu().numpy()

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(vis_img)
            axs[0].set_title("Image")
            axs[0].axis("off")

            axs[1].imshow(vis_mask, cmap="gray")
            axs[1].set_title("Binary forest mask (1=forest)")
            axs[1].axis("off")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(Visualization skipped) {e}")

        print("[DONE] Dataset test passed.")

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
