#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security analysis visualization: cover vs stego with RGB channel histograms.

Layout — 2 image pairs per row, 2 rows:
  Cover | Stego | Cover | Stego      ← images with titles
  hist  | hist  | hist  | hist       ← RGB histograms
  Cover | Stego | Cover | Stego
  hist  | hist  | hist  | hist

Usage:
    /Users/dmitryhoma/anaconda3/envs/pytorch/bin/python3 visualize_histograms.py
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
SEED       = 3
OUTPUT     = os.path.join(REPO, "edge_aspp_histograms.png")
# Explicit image indices into sorted val list (0-based).
# Change first 3 freely; index 16 = 0817.png (kept as image 4).
IMAGE_INDICES = [35, 22, 55, 16]

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model …")
model = ModelCheckpoint.load(CHECKPOINT, gpu=False, verbose=False)
model.encoder.eval()

# ── Pick images ───────────────────────────────────────────────────────────────
all_files = sorted([
    os.path.join(VAL_DIR, f)
    for f in os.listdir(VAL_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
chosen = [all_files[i] for i in IMAGE_INDICES]

preprocess = transforms.Compose([
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
])

# ── Encode all images ─────────────────────────────────────────────────────────
pairs = []   # list of (cover_np, stego_np)  uint8 H×W×3
with torch.no_grad():
    for i, path in enumerate(chosen):
        pil   = Image.open(path).convert("RGB")
        cover = preprocess(pil).unsqueeze(0)
        torch.manual_seed(SEED + i)
        msg   = torch.randint(0, 2, (1, DATA_DEPTH, CROP_SIZE, CROP_SIZE),
                              dtype=torch.float32)
        stego = model.encoder(cover, msg).clamp(0, 1)

        cover_np = (cover.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        stego_np = (stego.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pairs.append((cover_np, stego_np))
        print(f"  encoded {os.path.basename(path)}")

# ── Plot helpers ──────────────────────────────────────────────────────────────
CH_COLORS  = ["red", "green", "blue"]
CH_LABELS  = ["RED Channel", "GREEN Channel", "BLUE Channel"]
BINS       = 256
HIST_RANGE = (0, 255)


def plot_image(ax, img_np, title=None):
    ax.imshow(img_np)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, pad=5)


def plot_histogram(ax, img_np, legend_loc="upper right", legend_x=0.90):
    for ch, (color, label) in enumerate(zip(CH_COLORS, CH_LABELS)):
        counts, edges = np.histogram(img_np[..., ch].ravel(), bins=BINS,
                                     range=HIST_RANGE)
        ax.plot(edges[:-1], counts, color=color, label=label, linewidth=0.9)
    if legend_loc == "upper left":
        ax.legend(fontsize=5.5, loc="upper left", framealpha=0.7,
                  handlelength=1.2, handletextpad=0.4, borderpad=0.4,
                  labelspacing=0.25)
    else:
        ax.legend(fontsize=5.5, bbox_to_anchor=(legend_x, 1.0), loc="upper right",
                  framealpha=0.7, handlelength=1.2, handletextpad=0.4,
                  borderpad=0.4, labelspacing=0.25)
    ax.set_xlim(0, 250)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Build figure ──────────────────────────────────────────────────────────────
# Two separate GridSpecs — one per visual row — so the gap between them
# can be set independently from the tight hspace inside each half.
import matplotlib.gridspec as gridspec

PAIRS_PER_ROW = 2
fig = plt.figure(figsize=(16, 10.8))

# Each pair occupies a 2-column subgridspec: image(wide) | image(wide)
# The outer GridSpec has 4 columns with equal width for images,
# but each pair's histogram row is narrower via a nested subgridspec.
# Simpler: use outer 4-col gs for images, inner per-pair gs for hists
# with width_ratios to squeeze histogram width relative to image width.

# Outer gs: 2 rows (img, hist) × 4 cols, but histogram cols narrower
# Achieved with width_ratios: repeat [1, 0.75] per pair
W = [1, 0.75, 1, 0.75]   # col widths: img wide, hist narrow, img wide, hist narrow
# But cover+stego share the same column — keep equal; instead shrink overall
# by reducing figure width. For independent widths use nested gs per pair.

# Use one gs per half, 2 rows × 4 cols, uniform width (images determine width)
# Histogram width is controlled by the shared column — just reduce figure width.
gs_kw = dict(hspace=0.07, wspace=0.06, height_ratios=[2.8, 1.6])
gs_top = gridspec.GridSpec(2, 4, figure=fig, top=1.00, bottom=0.53, **gs_kw)
gs_bot = gridspec.GridSpec(2, 4, figure=fig, top=0.47, bottom=0.00, **gs_kw)
gs_halves = [gs_top, gs_bot]

last_pair_idx = len(pairs) - 1

for vrow, gs in enumerate(gs_halves):
    for pcol in range(PAIRS_PER_ROW):
        pair_idx  = vrow * PAIRS_PER_ROW + pcol
        cover_np, stego_np = pairs[pair_idx]
        col_cover = pcol * 2
        col_stego = pcol * 2 + 1
        is_last   = (pair_idx == last_pair_idx)

        ax_cov_img  = fig.add_subplot(gs[0, col_cover])
        ax_stg_img  = fig.add_subplot(gs[0, col_stego])
        ax_cov_hist = fig.add_subplot(gs[1, col_cover])
        ax_stg_hist = fig.add_subplot(gs[1, col_stego])

        plot_image(ax_cov_img,  cover_np, "Cover Image")
        plot_image(ax_stg_img,  stego_np, "Stego Image")

        loc = "upper left" if is_last else "upper right"
        plot_histogram(ax_cov_hist, cover_np, legend_loc=loc)
        plot_histogram(ax_stg_hist, stego_np, legend_loc=loc)

plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUTPUT}")
