import numpy as np
import torch
import torchvision
from torchvision import transforms


_DEFAULT_MU = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]


class TransformBuilder:
    """Builds image transformations for training and validation."""

    @staticmethod
    def default_transform(crop_size=360):
        """Create default training transform."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
        ])

    @staticmethod
    def validation_transform(crop_size=360):
        """Create validation transform without random augmentation."""
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
        ])


DEFAULT_TRANSFORM = TransformBuilder.default_transform()


class ImageFolder(torchvision.datasets.ImageFolder):
    """Custom ImageFolder with optional limit on dataset size."""

    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


class DataLoader(torch.utils.data.DataLoader):
    """Custom DataLoader for steganography training."""

    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=8, batch_size=4, *args, **kwargs):
        if transform is None:
            transform = DEFAULT_TRANSFORM

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )


class DataLoaderFactory:
    """Factory for creating train and validation data loaders."""

    @staticmethod
    def create_loaders(train_path, val_path, batch_size=4, num_workers=8,
                      train_limit=np.inf, val_limit=np.inf):
        """Create training and validation data loaders."""
        train_loader = DataLoader(
            train_path,
            transform=TransformBuilder.default_transform(),
            limit=train_limit,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_path,
            transform=TransformBuilder.validation_transform(),
            limit=val_limit,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader
