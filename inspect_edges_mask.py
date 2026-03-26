#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise what the learned EdgeNet is doing.

Outputs (per image):
  <out>/<stem>_cover.png            — original cover image
  <out>/<stem>_overlay.png          — edge_mask heatmap alpha-blended over cover
                                      (shows WHERE the encoder concentrates perturbations)
  <out>/<stem>_mask_comparison.png  — side-by-side: Cover | Edge Mask (learned) | Sobel

Notes
-----
The overlay uses `edge_mask = ε + (1−ε)·edge_map` (not the raw EdgeNet output),
because `edge_mask` is what literally multiplies each perturbation δ_t inside the
ConvGRU loop.  Red intensity ∝ embedding weight at that pixel.

The Sobel comparison panel lets you see whether the learned mask tracks classical
gradient edges or has diverged to encode steganography-specific structure.

Usage
-----
python inspect_edges.py --image PATH --checkpoint PATH [--out edge_vis] [--gpu]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imread


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load image → (1, 3, H, W) float32 in [-1, 1]."""
    arr = imread(path, pilmode="RGB").astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def sobel_edges(cover: torch.Tensor) -> torch.Tensor:
    """Classical Sobel magnitude on grayscale → (1, 1, H, W) in [0, 1]."""
    w   = cover.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    lum = (cover * w).sum(1, keepdim=True)
    kx  = lum.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky  = lum.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    mag = (F.conv2d(lum, kx, padding=1).pow(2) +
           F.conv2d(lum, ky, padding=1).pow(2)).sqrt()
    return mag / mag.max().clamp(min=1e-8)


def to_np_rgb(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) in [-1, 1] → H×W×3 uint8."""
    return ((t.squeeze(0).clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def to_np_gray(t: torch.Tensor) -> np.ndarray:
    """(1, 1, H, W) in [0, 1] → H×W float32."""
    return t.squeeze().clamp(0, 1).cpu().numpy()


def make_overlay(cover: torch.Tensor, edge_mask: torch.Tensor) -> np.ndarray:
    """
    Alpha-blend a 'plasma' colormap heatmap over the cover image.

    edge_mask ∈ [ε, 1.0] — the actual per-pixel embedding multiplier.
    High mask → strong heatmap tint; low mask → cover shows through.
    'plasma' runs dark-purple → blue → yellow (no red).
    """
    import matplotlib.cm as cm
    alpha = 0.50
    rgb   = ((cover.squeeze(0).clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()  # H W 3 [0,1]
    mask  = edge_mask.squeeze().clamp(0, 1).cpu().numpy()                               # H W   [0,1]

    # Normalise mask to [0,1] for colormap (min–max per image so full range is used)
    mask_n  = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    heat    = cm.plasma(mask_n)[..., :3]   # H W 3, plasma colormap RGB

    out = rgb * (1.0 - alpha) + heat * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _stats(t: torch.Tensor, name: str) -> None:
    print(f"  {name:<22} min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

REPO       = os.path.dirname(os.path.abspath(__file__))
IMAGE      = os.path.expanduser("~/Projects/datasets/div2k/val/_/0806.png")
CHECKPOINT = os.path.join(REPO, "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/25.rsbpp-0.973312.p")
OUT_DIR    = os.path.join(REPO, "edge_vis")


def main() -> None:
    device = torch.device("cpu")
    print(f"Device : {device}")

    # ── Load encoder ──────────────────────────────────────────────────────────
    from steganogan.utils.checkpoint import ModelCheckpoint
    model   = ModelCheckpoint.load(CHECKPOINT, gpu=False, verbose=True)
    encoder = model.encoder.to(device).eval()

    # ── Load image ────────────────────────────────────────────────────────────
    cover = load_image(IMAGE, device)
    _, _, H, W = cover.shape
    print(f"Image  : {IMAGE}  ({W}×{H})\n")

    # ── Run EdgeNet ───────────────────────────────────────────────────────────
    with torch.no_grad():
        edge_map  = encoder.edge_net(cover)
        edge_mask = encoder.edge_epsilon + (1.0 - encoder.edge_epsilon) * edge_map
        sobel     = sobel_edges(cover)

    print("Tensor stats:")
    _stats(edge_map,  "edge_map (raw EdgeNet)")
    _stats(edge_mask, "edge_mask (with ε floor)")
    _stats(sobel,     "sobel (classical)")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(IMAGE))[0]

    # ── Cover ─────────────────────────────────────────────────────────────────
    cover_np = to_np_rgb(cover)
    Image.fromarray(cover_np).save(f"{OUT_DIR}/{stem}_cover.png")

    # ── Mask comparison (single figure) ───────────────────────────────────────
    mask_np  = to_np_gray(edge_mask)
    sobel_np = to_np_gray(sobel)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.04})

    panels = [
        (cover_np, "Cover Image",    None),
        (mask_np,  "Edge-ASPP Mask", "gray"),
        (sobel_np, "Sobel Mask",     "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12, pad=6)
        ax.axis("off")

    plt.savefig(f"{OUT_DIR}/{stem}_mask_comparison.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  {stem}_cover.png            — original")
    print(f"  {stem}_mask_comparison.png  — Cover | Edge-ASPP Mask | Sobel Mask")


if __name__ == "__main__":
    main()
