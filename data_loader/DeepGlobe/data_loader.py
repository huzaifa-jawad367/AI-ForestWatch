from torch.utils.data import DataLoader
from torchvision import transforms
from deepglobe_forest_dataset import DeepGlobeForestDataset


def get_deepglobe_loaders(
    root: str = "data",
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Returns train_loader and val_loader for the binary forest segmentation task
    using the DeepGlobe dataset.
    """
    common_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = DeepGlobeForestDataset(
        root=root,
        split="train",
        transforms=common_transforms,
        checksum=False,
    )
    val_ds = DeepGlobeForestDataset(
        root=root,
        split="test",
        transforms=common_transforms,
        checksum=False,
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
    train_loader, val_loader = get_deepglobe_loaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    imgs, masks = next(iter(train_loader))
    print("Images:", imgs.shape)
    print("Masks:", masks.shape)
