from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import LoveDASegmentationDataset


def get_train_val_loaders(
    root: str = "data",
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Returns:
        train_loader, val_loader: two DataLoader instances for training and validation.
    """
    # Example transforms — adapt as needed for your SegFormer pipeline
    common_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = LoveDASegmentationDataset(
        root=root,
        split="train",
        scene=("urban", "rural"),
        transforms=common_transforms,
        download=True,
    )
    val_ds = LoveDASegmentationDataset(
        root=root,
        split="val",
        scene=("urban", "rural"),
        transforms=common_transforms,
        download=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_train_val_loaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    # Peek a single batch
    imgs, masks = next(iter(train_loader))
    print("Images:", imgs.shape)
    print("Masks:", masks.shape)
