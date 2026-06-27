# -*- coding: utf-8 -*-
"""
SteganoGAN v3 dataset loader for steganalysis.

Differs from `SteganoganDataset` (v1/v2) in how splits are obtained: v3 ships
explicit, reproducible split manifests produced by the dataset generator, so we
do NOT re-derive a random split here. Reading the manifests preserves the
generator's guarantees — splitting by image id (cover and its stego stay
together) and stratification by encoder architecture.

Dataset structure
-----------------
    steganogan-dataset-v3/
        cover/      *.png   — cover images, 512×512 (label 0)
        basic/      *.png   — BasicEncoder stego images    (label 1)
        dense/      *.png   — DenseEncoder stego images    (label 1)
        residual/   *.png   — ResidualEncoder stego images (label 1)
        splits/     train.txt | val.txt | test.txt  — one filename per line
        metadata.csv        — filename, architecture, split, payload_seed,
                              payload_bits_per_pixel

Each cover has exactly ONE stego, in the folder of its randomly assigned
architecture (so classes are balanced 1:1 within every split). The
architecture of each stego is read from metadata.csv.

Usage
-----
    ds = SteganoganV3Dataset(root="~/datasets/steganogan-dataset-v3", split="train")
    img, label = ds[0]
"""

import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# ── Supported stego variants ──────────────────────────────────────────────────

STEGO_DIRS: Tuple[str, ...] = ("basic", "dense", "residual")
COVER_DIR:  str             = "cover"
SPLITS_DIR: str             = "splits"
METADATA:   str             = "metadata.csv"


# ── Dataset ───────────────────────────────────────────────────────────────────

class SteganoganV3Dataset(Dataset):
    """
    SteganoGAN v3 binary steganalysis dataset (cover vs stego).

    Splits come from the on-disk manifests, not from an RNG. ``val_frac`` /
    ``test_frac`` / ``seed`` are accepted for signature parity with the v1/v2
    loader but are ignored (the manifest is authoritative); ``seed`` is still
    used to shuffle when ``max_samples`` caps the sample list.

    Parameters
    ----------
    root        : root directory containing cover/, basic/, dense/, residual/,
                  splits/, metadata.csv
    split       : "train" | "val" | "test"
    transform   : torchvision transform applied to every image
    stego_algs  : which stego variants to include (default: all three)
    max_samples : hard cap on total samples (useful for quick experiments)
    """

    LABEL_COVER: int = 0
    LABEL_STEGO: int = 1

    def __init__(
        self,
        root:        str,
        split:       str                    = "train",
        transform:   Optional[Callable]     = None,
        stego_algs:  Sequence[str]          = STEGO_DIRS,
        val_frac:    float                  = 0.1,   # ignored (manifest wins)
        test_frac:   float                  = 0.1,   # ignored (manifest wins)
        seed:        int                    = 42,
        max_samples: Optional[int]          = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"
        self.root      = Path(root).expanduser()
        self.split     = split
        self.transform = transform

        cover_dir = self.root / COVER_DIR
        arch_of   = self._read_architectures()             # filename -> architecture
        split_files = self._read_split_manifest(split)     # filenames in this split

        stego_algs = set(stego_algs)

        # Build sample list: (path, label). One cover + its single stego per id.
        samples: List[Tuple[Path, int]] = []
        for fname in split_files:
            arch = arch_of.get(fname)
            if arch is None or arch not in stego_algs:
                # Skip ids whose architecture is filtered out (keeps pairs intact).
                continue

            cover_path = cover_dir / fname
            stego_path = self.root / arch / fname
            if cover_path.exists():
                samples.append((cover_path, self.LABEL_COVER))
            if stego_path.exists():
                samples.append((stego_path, self.LABEL_STEGO))

        if not samples:
            raise FileNotFoundError(
                f"No samples for split={split!r} under {self.root}. "
                "Check that the v3 dataset (with splits/ and metadata.csv) is "
                "unpacked correctly."
            )

        if max_samples is not None:
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:max_samples]

        self.samples = samples
        self.architecture_of = arch_of

    # ── Manifest / metadata readers ───────────────────────────────────────────

    def _read_architectures(self) -> Dict[str, str]:
        """Map every filename to its encoder architecture via metadata.csv."""
        meta_path = self.root / METADATA
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing {meta_path}. The v3 dataset must ship metadata.csv "
                "(filename, architecture, split, ...)."
            )
        arch_of: Dict[str, str] = {}
        with open(meta_path, newline="") as f:
            for row in csv.DictReader(f):
                arch_of[row["filename"]] = row["architecture"]
        return arch_of

    def _read_split_manifest(self, split: str) -> List[str]:
        """Read filenames for *split* from splits/<split>.txt."""
        manifest = self.root / SPLITS_DIR / f"{split}.txt"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Missing split manifest {manifest}. Expected "
                f"{SPLITS_DIR}/train.txt, val.txt, test.txt."
            )
        with open(manifest) as f:
            return [line.strip() for line in f if line.strip()]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def num_cover(self) -> int:
        return sum(1 for _, l in self.samples if l == self.LABEL_COVER)

    @property
    def num_stego(self) -> int:
        return sum(1 for _, l in self.samples if l == self.LABEL_STEGO)

    def __repr__(self) -> str:
        return (
            f"SteganoganV3Dataset(split={self.split!r}, "
            f"cover={self.num_cover}, stego={self.num_stego}, "
            f"total={len(self)})"
        )
