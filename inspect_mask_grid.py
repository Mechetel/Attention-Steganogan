#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise EdgeNet mask in a 2-row grid layout.

Layout
------
  Row 1 (full width) : Cover Image
  Row 2 (two panels) : Edge-ASPP Mask  |  Sobel Mask

Output
------
  <out>/<stem>_mask_grid.png   — 2-row grid figure

Skips images whose output file already exists.

Usage
-----
python inspect_mask_grid.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio
from imageio.v2 import imread


# ── Config ────────────────────────────────────────────────────────────────────

REPO       = os.path.dirname(os.path.abspath(__file__))
IMAGES     = [
    os.path.expanduser("~/Projects/datasets/div2k/val/_/0806.png"),
]
CHECKPOINT = os.path.join(REPO, "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/25.rsbpp-0.973312.p")
OUT_DIR    = os.path.join(REPO, "edge_vis")


# ── Helpers (same as inspect_edges_mask.py) ───────────────────────────────────

def load_image(path: str, device: torch.device) -> torch.Tensor:
    arr = imread(path, pilmode="RGB").astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def sobel_edges(cover: torch.Tensor) -> torch.Tensor:
    w   = cover.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    lum = (cover * w).sum(1, keepdim=True)
    kx  = lum.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky  = lum.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    mag = (F.conv2d(lum, kx, padding=1).pow(2) +
           F.conv2d(lum, ky, padding=1).pow(2)).sqrt()
    return mag / mag.max().clamp(min=1e-8)


def to_np_rgb(t: torch.Tensor) -> np.ndarray:
    return ((t.squeeze(0).clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def to_np_gray(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().clamp(0, 1).cpu().numpy()


# ── Grid figure ───────────────────────────────────────────────────────────────

def save_mask_grid(cover_np, mask_np, sobel_np, out_path: str) -> None:
    fig = plt.figure(figsize=(13, 9))
    gs  = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.04)

    ax_cover = fig.add_subplot(gs[0, :])   # row 0, spans both columns
    ax_mask  = fig.add_subplot(gs[1, 0])   # row 1, left
    ax_sobel = fig.add_subplot(gs[1, 1])   # row 1, right

    ax_cover.imshow(cover_np)
    ax_cover.set_title("Cover Image", fontsize=20, pad=10)
    ax_cover.axis("off")

    ax_mask.imshow(mask_np, cmap="gray")
    ax_mask.set_title("Edge-ASPP Mask", fontsize=20, pad=10)
    ax_mask.axis("off")

    ax_sobel.imshow(sobel_np, cmap="gray")
    ax_sobel.set_title("Sobel Mask", fontsize=20, pad=10)
    ax_sobel.axis("off")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    from steganogan.utils.checkpoint import ModelCheckpoint
    model   = ModelCheckpoint.load(CHECKPOINT, gpu=False, verbose=True)
    encoder = model.encoder.to(device).eval()

    for image_path in IMAGES:
        stem     = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUT_DIR, f"{stem}_mask_grid.png")

        if os.path.exists(out_path):
            print(f"Skip (exists) : {out_path}")
            continue

        print(f"Processing    : {image_path}")
        cover = load_image(image_path, device)

        with torch.no_grad():
            edge_map  = encoder.edge_net(cover)
            edge_mask = encoder.edge_epsilon + (1.0 - encoder.edge_epsilon) * edge_map
            sobel     = sobel_edges(cover)

        save_mask_grid(
            cover_np  = to_np_rgb(cover),
            mask_np   = to_np_gray(edge_mask),
            sobel_np  = to_np_gray(sobel),
            out_path  = out_path,
        )
        print(f"Saved         : {out_path}")


if __name__ == "__main__":
    main()
