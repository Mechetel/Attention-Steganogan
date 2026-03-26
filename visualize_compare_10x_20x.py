#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method comparison visualization: LSB vs dense vs edge_aspp on one image.

Layout (rows = methods, cols = Cover | Stego | Residual ×10 | Residual ×20):

  LSB        | cover | stego | res×10 | res×20
  dense      | cover | stego | res×10 | res×20
  edge_aspp  | cover | stego | res×10 | res×20

Usage:
    /Users/dmitryhoma/anaconda3/envs/pytorch/bin/python3 visualize_compare.py
"""

import os
import sys
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from steganogan.utils.checkpoint import ModelCheckpoint

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "dense": os.path.join(
        REPO, "models/div2k/dense_dense/1773340283/32.rsbpp-0.965443.p"
    ),
    "edge\\_unet": os.path.join(
        REPO, "models/div2k/edge_unet_dense/1773386504/32.rsbpp-0.969229.p"
    ),
    "edge\\_aspp": os.path.join(
        REPO, "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/25.rsbpp-0.973312.p"
    ),
}

VAL_DIR    = os.path.expanduser("~/Projects/datasets/div2k/val/_")
CROP_SIZE  = 360
DATA_DEPTH = 1
SEED       = 12
IMAGE_IDX  = 5      # index into sorted val list (change to pick different image)
OUTPUT     = os.path.join(REPO, "comparision_10x_20x.png")


# ── LSB steganography ─────────────────────────────────────────────────────────
def lsb_encode(cover: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    """Replace LSB of each 8-bit channel pixel with one message bit."""
    cover_uint8 = (cover.squeeze(0) * 255).round().to(torch.uint8)  # (3,H,W)
    msg_bits    = msg.squeeze(0).squeeze(0).to(torch.uint8)          # (H,W) values 0/1
    # broadcast same bit plane across all 3 channels
    stego_uint8 = (cover_uint8 & 0xFE) | msg_bits.unsqueeze(0)
    return (stego_uint8.float() / 255.0).unsqueeze(0)


# ── Residual display (white background) ───────────────────────────────────────
def residual_vis(cover_np: np.ndarray, stego_np: np.ndarray,
                 scale: float) -> np.ndarray:
    diff = stego_np - cover_np
    return np.clip(1.0 - np.abs(diff) * scale, 0, 1)


# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models …")
models = {}
for name, ckpt in CHECKPOINTS.items():
    m = ModelCheckpoint.load(ckpt, gpu=False, verbose=False)
    m.encoder.eval()
    models[name] = m
    print(f"  {name} loaded")

# ── Load one cover image ──────────────────────────────────────────────────────
all_files = sorted([
    os.path.join(VAL_DIR, f)
    for f in os.listdir(VAL_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
img_path = all_files[IMAGE_IDX]
print(f"\nUsing image: {os.path.basename(img_path)}")

preprocess = transforms.Compose([
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
])
pil   = Image.open(img_path).convert("RGB")
cover = preprocess(pil).unsqueeze(0)   # (1,3,H,W)

torch.manual_seed(SEED)
msg = torch.randint(0, 2, (1, DATA_DEPTH, CROP_SIZE, CROP_SIZE), dtype=torch.float32)

# ── Encode with each method ───────────────────────────────────────────────────
cover_np = cover.squeeze(0).permute(1, 2, 0).numpy()

stegos = {}
stegos["LSB"] = lsb_encode(cover, msg).squeeze(0).permute(1, 2, 0).numpy()

with torch.no_grad():
    for name, m in models.items():
        stego = m.encoder(cover, msg).clamp(0, 1)
        stegos[name] = stego.squeeze(0).permute(1, 2, 0).numpy()

# ── Build figure ──────────────────────────────────────────────────────────────
method_names = ["LSB"] + list(CHECKPOINTS.keys())
n_rows = len(method_names)
n_cols = 4   # Cover | Stego | Res×10 | Res×20

col_titles = ["Cover Image", "Stego Image", "Residual ×10", "Residual ×20"]
row_labels  = ["LSB", "dense", "edge_unet", "edge_aspp"]

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(15, n_rows * 3.8),
    gridspec_kw={"wspace": 0.01, "hspace": 0.03},
)

# Column headers
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=13, pad=8)

for row, name in enumerate(method_names):
    stego_np = stegos[name]
    cols = [
        cover_np,
        stego_np,
        residual_vis(cover_np, stego_np, scale=10),
        residual_vis(cover_np, stego_np, scale=20),
    ]
    for col, img in enumerate(cols):
        ax = axes[row, col]
        ax.imshow(img)
        ax.axis("off")
        if col >= 2:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#bbbbbb")
                spine.set_linewidth(0.8)

    # Row label on the left
    axes[row, 0].set_ylabel(row_labels[row], fontsize=12, rotation=90,
                             labelpad=6, va="center")
    axes[row, 0].axis("off")   # re-hide after ylabel

plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUTPUT}")
