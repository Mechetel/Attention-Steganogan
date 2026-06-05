#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of cover, stego, and residual images for edge_aspp.

Produces a figure with 4 rows × 4 columns:
  Cover Image | Stego Image | Residual ×1 | Residual ×10

Usage:
    /Users/dmitryhoma/anaconda3/envs/pytorch/bin/python3 visualize_stego.py
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
CHECKPOINT = os.path.join(
    REPO, "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/25.rsbpp-0.973312.p"
)
VAL_DIR    = os.path.expanduser("~/Projects/datasets/div2k/val/_")
CROP_SIZE  = 360
DATA_DEPTH = 1
NUM_ROWS   = 4
SEED       = 7
OUTPUT     = os.path.join(REPO, "edge_aspp_visualization.png")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model …")
model = ModelCheckpoint.load(CHECKPOINT, gpu=False, verbose=False)
model.encoder.eval()

# ── Select images ─────────────────────────────────────────────────────────────
random.seed(SEED)
all_files = sorted([
    os.path.join(VAL_DIR, f)
    for f in os.listdir(VAL_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
chosen = random.sample(all_files, NUM_ROWS)

preprocess = transforms.Compose([
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
])

# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    NUM_ROWS, 4,
    figsize=(14, NUM_ROWS * 3.6),
    gridspec_kw={"wspace": 0.04, "hspace": 0.06},
)

col_titles = ["Cover Image", "Stego Image", "Residual ×1", "Residual ×10"]
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=13, fontweight="normal", pad=8)

with torch.no_grad():
    for row, img_path in enumerate(chosen):
        pil   = Image.open(img_path).convert("RGB")
        cover = preprocess(pil).unsqueeze(0)          # (1,3,H,W)

        torch.manual_seed(SEED + row)
        msg   = torch.randint(0, 2, (1, DATA_DEPTH, CROP_SIZE, CROP_SIZE),
                              dtype=torch.float32)
        stego = model.encoder(cover, msg).clamp(0, 1)  # (1,3,H,W)

        cover_np = cover.squeeze(0).permute(1, 2, 0).numpy()
        stego_np = stego.squeeze(0).permute(1, 2, 0).numpy()

        residual     = stego_np - cover_np                      # per-channel diff
        # White background: 1.0 = no difference, colors appear where diff exists
        residual_x1  = np.clip(1.0 - np.abs(residual),        0, 1)
        residual_x10 = np.clip(1.0 - np.abs(residual) * 10.0, 0, 1)

        for col, img in enumerate([cover_np, stego_np, residual_x1, residual_x10]):
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            # thin border for residual columns (matches reference style)
            if col >= 2:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#bbbbbb")
                    spine.set_linewidth(0.8)

plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUTPUT}")
