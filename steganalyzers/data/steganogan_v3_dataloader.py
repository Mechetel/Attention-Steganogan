# -*- coding: utf-8 -*-
"""
DataLoader factory for the SteganoGAN v3 steganalysis dataset.

Mirrors `SteganoganDataLoaderFactory` but binds to `SteganoganV3Dataset`, which
takes its train/val/test splits from the on-disk manifests instead of an RNG.
The ``val_frac`` / ``test_frac`` / ``seed`` arguments are kept for call-site
compatibility with the other factories but do not affect the split.
"""

from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .steganogan_v3 import SteganoganV3Dataset, STEGO_DIRS


# ── Default transforms ─────────────────────────────────────────────────────────

def _train_transform(crop_size: int) -> transforms.Compose:
    """Augmented pipeline: random crop + flip + tensor in [0,1]."""
    return transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # Pixel values kept in [0, 1] — HPF layers are sensitive to absolute
        # magnitudes, so no mean/std normalisation here.
    ])


def _eval_transform(crop_size: int) -> transforms.Compose:
    """Minimal pipeline: centre crop + tensor."""
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


# ── Factory ────────────────────────────────────────────────────────────────────

class SteganoganV3DataLoaderFactory:
    """Creates train / val / test DataLoaders for the SteganoGAN v3 dataset."""

    @staticmethod
    def create(
        root:         str,
        batch_size:   int             = 32,
        num_workers:  int             = 4,
        crop_size:    int             = 256,
        stego_algs:   Sequence[str]   = STEGO_DIRS,
        val_frac:     float           = 0.1,   # ignored (manifest wins)
        test_frac:    float           = 0.1,   # ignored (manifest wins)
        seed:         int             = 42,
        balanced:     bool            = True,
        max_samples:  Optional[int]   = None,
        pin_memory:   bool            = True,
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds = SteganoganV3Dataset(
            root=root, split="train",
            transform=_train_transform(crop_size),
            stego_algs=stego_algs,
            seed=seed, max_samples=max_samples,
        )
        val_ds = SteganoganV3Dataset(
            root=root, split="val",
            transform=_eval_transform(crop_size),
            stego_algs=stego_algs,
            seed=seed,
        )

        train_sampler = (
            SteganoganV3DataLoaderFactory._balanced_sampler(train_ds)
            if balanced else None
        )
        shuffle_train = (train_sampler is None)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader

    @staticmethod
    def create_test(
        root:        str,
        batch_size:  int           = 32,
        num_workers: int           = 4,
        crop_size:   int           = 256,
        stego_algs:  Sequence[str] = STEGO_DIRS,
        val_frac:    float         = 0.1,   # ignored (manifest wins)
        test_frac:   float         = 0.1,   # ignored (manifest wins)
        seed:        int           = 42,
        pin_memory:  bool          = True,
    ) -> DataLoader:
        test_ds = SteganoganV3Dataset(
            root=root, split="test",
            transform=_eval_transform(crop_size),
            stego_algs=stego_algs,
            seed=seed,
        )
        return DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    @staticmethod
    def _balanced_sampler(dataset: SteganoganV3Dataset) -> WeightedRandomSampler:
        """Equalise cover / stego class frequencies per batch."""
        labels = torch.tensor([lbl for _, lbl in dataset.samples])
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
