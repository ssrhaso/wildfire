from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CSV_PATH = Path("data/processed/labels.csv")


def get_train_transform() -> v2.Compose:
    """Training transform with augmentation and ImageNet normalisation."""
    return v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        v2.RandomGrayscale(p=0.05),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
        v2.ToTensor(),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_eval_transform() -> v2.Compose:
    """Deterministic transform for validation and test splits."""
    return v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class WildfireDataset(Dataset):
    """PyTorch Dataset for the wildfire binary classification task."""

    def __init__(
        self,
        csv_path: Path = CSV_PATH,
        split: str = "train",
        transform: Optional[v2.Compose] = None,
    ) -> None:
        """Load labels.csv, filter by split, and assign the appropriate transform."""
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transform() if split == "train" else get_eval_transform()

        fire = (self.df["label"] == 1).sum()
        nofire = (self.df["label"] == 0).sum()
        print(f"  [{split}] fire: {fire}, nofire: {nofire}, total: {len(self.df)}")

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load an image, apply transforms, and return (tensor, label)."""
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["path"]).convert("RGB")
        except Exception as e:
            print(f"  Error loading {row['path']}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        image = self.transform(image)
        return image, int(row["label"])


def get_dataloaders(
    csv_path: Path = CSV_PATH,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders with standard configuration."""
    train_ds = WildfireDataset(csv_path, split="train")
    val_ds = WildfireDataset(csv_path, split="val")
    test_ds = WildfireDataset(csv_path, split="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, test_loader


def compute_class_weights(dataset: WildfireDataset) -> torch.FloatTensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    labels = dataset.df["label"].values
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2).float()
    weights = len(labels) / (2.0 * counts)
    print(f"  Class weights — nofire (0): {weights[0]:.4f}, fire (1): {weights[1]:.4f}")
    return weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=0)

    weights = compute_class_weights(train_loader.dataset)

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        images, labels = next(iter(loader))
        fire = (labels == 1).sum().item()
        nofire = (labels == 0).sum().item()
        print(f"\n  [{name}] batch shape: {images.shape}")
        print(f"  [{name}] labels — fire: {fire}, nofire: {nofire}")
        print(f"  [{name}] pixel min: {images.min():.4f}, max: {images.max():.4f}, mean: {images.mean():.4f}")

