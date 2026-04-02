#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a stego-image dataset for steganalysis evaluation.

Picks 1000 random MSCOCO images, creates D=1 stego images with three models:
  - SteganoGAN Dense
  - Edge-Guided U-Net
  - Edge-Aware DenseASPP

Output layout:
  <OUT_DIR>/cover/              — 1000 cover images (PNG)
  <OUT_DIR>/steganogan-dense/   — 1000 stego (Dense encoder)
  <OUT_DIR>/edge-unet/          — 1000 stego (Edge U-Net encoder)
  <OUT_DIR>/edge-aspp/          — 1000 stego (Edge ASPP encoder)

Usage:
    python generate_stego_dataset.py
"""

import os
import sys
import random

import torch
from PIL import Image
from tqdm import tqdm

# ── Ensure steganogan package is importable ──────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from steganogan.utils.checkpoint import ModelCheckpoint

# ── Config ───────────────────────────────────────────────────────────────────
MSCOCO_DIR = os.path.expanduser("~/Projects/datasets/mscoco/test2017")
OUT_DIR = os.path.expanduser("~/Projects/datasets/khoma-stego-images")

MODELS = {
    "steganogan-dense": os.path.join(
        REPO, "models/div2k/dense_dense/1773340283/32.rsbpp-0.965443.p"
    ),
    "edge-unet": os.path.join(
        REPO, "models/div2k/edge_unet_dense/1773386504/32.rsbpp-0.969229.p"
    ),
    "edge-aspp": os.path.join(
        REPO, "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/25.rsbpp-0.973312.p"
    ),
}

NUM_IMAGES = 1000
SEED = 42
MESSAGE = "Khoma PhD steganalysis evaluation dataset"


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Collect and sample images ────────────────────────────────────────────
    all_images = sorted([
        f for f in os.listdir(MSCOCO_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    selected = random.sample(all_images, NUM_IMAGES)
    print(f"Selected {len(selected)} images from MSCOCO test2017")

    # ── Create output directories ────────────────────────────────────────────
    cover_dir = os.path.join(OUT_DIR, "cover")
    os.makedirs(cover_dir, exist_ok=True)
    model_dirs = {}
    for name in MODELS:
        d = os.path.join(OUT_DIR, name)
        os.makedirs(d, exist_ok=True)
        model_dirs[name] = d

    # ── Save cover images as PNG ─────────────────────────────────────────────
    print("Saving cover images …")
    cover_paths = []
    filenames = []
    for i, fname in enumerate(tqdm(selected, desc="Cover")):
        src_path = os.path.join(MSCOCO_DIR, fname)
        out_name = f"{i:04d}.png"
        filenames.append(out_name)
        cover_path = os.path.join(cover_dir, out_name)
        cover_paths.append(cover_path)
        Image.open(src_path).convert("RGB").save(cover_path)

    # ── Generate stego images per model ──────────────────────────────────────
    for model_name, ckpt_path in MODELS.items():
        print(f"\nLoading {model_name} …")
        model = ModelCheckpoint.load(
            ckpt_path, gpu=torch.cuda.is_available(), verbose=False
        )
        out_model_dir = model_dirs[model_name]

        for i in tqdm(range(NUM_IMAGES), desc=model_name):
            stego_path = os.path.join(out_model_dir, filenames[i])
            model.encode(cover_paths[i], stego_path, MESSAGE)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone! Output in {OUT_DIR}")
    print(f"  cover/             — {NUM_IMAGES} images")
    for name in MODELS:
        print(f"  {name + '/':21s}— {NUM_IMAGES} images")


if __name__ == "__main__":
    main()
