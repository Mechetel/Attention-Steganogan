import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader as TorchDataLoader
from typing import Optional, Tuple


_DEFAULT_MU    = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]


class TransformBuilder:
    """Builds image transformations for training and validation."""

    @staticmethod
    def default_transform(crop_size: int = 360) -> transforms.Compose:
        """Create default training transform with random augmentation."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
        ])

    @staticmethod
    def validation_transform(crop_size: int = 360) -> transforms.Compose:
        """Create validation transform without random augmentation."""
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
        ])


DEFAULT_TRANSFORM = TransformBuilder.default_transform()


class ImageFolder(torchvision.datasets.ImageFolder):
    """Custom ImageFolder with optional limit on dataset size."""

    def __init__(self, path: str, transform: transforms.Compose,
                 limit: float = np.inf) -> None:
        super().__init__(path, transform=transform)
        self.limit: float = limit

    def __len__(self) -> int:
        return min(super().__len__(), int(self.limit))


class DataLoader(TorchDataLoader):
    """Custom DataLoader for steganography training."""

    def __init__(self, path: str, transform: Optional[transforms.Compose] = None,
                 limit: float = np.inf, shuffle: bool = True,
                 num_workers: int = 8, batch_size: int = 4,
                 *args, **kwargs) -> None:
        if transform is None:
            transform = DEFAULT_TRANSFORM

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs,
        )


class DataLoaderFactory:
    """Factory for creating train and validation data loaders."""

    @staticmethod
    def create_loaders(
        train_path: str,
        val_path: str,
        batch_size: int = 4,
        num_workers: int = 8,
        train_limit: float = np.inf,
        val_limit: float = np.inf,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create and return (train_loader, val_loader) pairs."""
        train_loader = DataLoader(
            train_path,
            transform=TransformBuilder.default_transform(),
            limit=train_limit,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_path,
            transform=TransformBuilder.validation_transform(),
            limit=val_limit,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader
