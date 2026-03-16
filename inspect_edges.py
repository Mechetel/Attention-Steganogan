#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise what the learned EdgeNet is doing.

Outputs (per image):
  <stem>_cover.png          — original cover image
  <stem>_edge_learned.png   — EdgeNet output  (brighter = stronger learned edge)
  <stem>_edge_mask.png      — mask applied to perturbations  (epsilon floor visible)
  <stem>_edge_sobel.png     — classical Sobel for comparison
  <stem>_overlay.png        — learned edges tinted red over cover

Usage
-----
# From a trained checkpoint:
python inspect_edges.py --image data/callback_images/foo.png \
                        --checkpoint models/edge_aspp_edge_aware_dense/1234/weights.steg

# Without a checkpoint (random weights — sanity check that the plumbing works):
python inspect_edges.py --image data/callback_images/foo.png

┌──────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│       File       │                                                What to look for                                                │
├──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ edge_learned.png │ Should highlight object contours. Compare with Sobel — if identical, the network hasn't diverged from          │
│                  │ classical edges; if different, it's learned steganography-specific features                                    │
├──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ edge_mask.png    │ Same but with the ε=0.05 floor baked in — this is literally what gets multiplied onto perturbations. Bright =  │
│                  │ more embedding                                                                                                 │
├──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ edge_sobel.png   │ Classical baseline for comparison                                                                              │
├──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ overlay.png      │ Red tint = where the encoder concentrates changes on the cover                                                 │
└──────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from imageio import imread


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load an image file → (1, 3, H, W) float32 in [-1, 1]."""
    arr = imread(path, pilmode="RGB").astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def sobel_edges(cover: torch.Tensor) -> torch.Tensor:
    """Classical Sobel magnitude on grayscale cover → (1, 1, H, W) in [0, 1]."""
    w   = cover.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    lum = (cover * w).sum(1, keepdim=True)
    kx  = lum.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky  = lum.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    mag = (F.conv2d(lum, kx, padding=1).pow(2) + F.conv2d(lum, ky, padding=1).pow(2)).sqrt()
    return mag / mag.max().clamp(min=1e-8)


def to_pil_gray(t: torch.Tensor) -> Image.Image:
    """(1, 1, H, W) in [0, 1] → PIL L image."""
    arr = (t.squeeze().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr, mode="L")


def to_pil_rgb(t: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) in [-1, 1] → PIL RGB image."""
    arr = ((t.squeeze(0).clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def overlay(cover: torch.Tensor, edge: torch.Tensor) -> Image.Image:
    """Tint high-edge pixels red over the cover image."""
    rgb = ((cover.squeeze(0).clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu()   # H W 3
    e   = edge.squeeze().clamp(0, 1).cpu()                                     # H W
    out = rgb.clone()
    out[..., 0] = (rgb[..., 0] + 0.6 * e).clamp(0, 1)   # push red up
    out[..., 1] = (rgb[..., 1] - 0.3 * e).clamp(0, 1)   # pull green down
    out[..., 2] = (rgb[..., 2] - 0.3 * e).clamp(0, 1)   # pull blue down
    return Image.fromarray((out * 255).byte().numpy())


def _stats(t: torch.Tensor, name: str) -> None:
    print(f"  {name:<18} min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise EdgeNet edge maps")
    parser.add_argument("--image",      required=True,        help="Cover image path (PNG/JPG)")
    parser.add_argument("--checkpoint", default=None,         help="Path to .steg checkpoint (optional)")
    parser.add_argument("--out",        default="edge_vis",   help="Output directory")
    parser.add_argument("--gpu",        action="store_true",  help="Use GPU/MPS if available")
    args = parser.parse_args()

    if args.gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    # ── Load encoder ──────────────────────────────────────────────────────────
    if args.checkpoint:
        from steganogan.utils.checkpoint import ModelCheckpoint
        model   = ModelCheckpoint.load(args.checkpoint, gpu=args.gpu, verbose=True)
        encoder = model.encoder
        print("Loaded trained encoder from checkpoint.")
    else:
        from steganogan.models.encoders.edge_aspp import EdgeAwareDenseASPPEncoder
        encoder = EdgeAwareDenseASPPEncoder(data_depth=1)
        print("No checkpoint — using randomly initialised encoder (sanity check).")

    encoder = encoder.to(device).eval()

    # ── Load image ────────────────────────────────────────────────────────────
    cover = load_image(args.image, device)
    _, _, H, W = cover.shape
    print(f"Image  : {args.image}  ({W}×{H})\n")

    # ── Run EdgeNet ───────────────────────────────────────────────────────────
    with torch.no_grad():
        edge_map  = encoder.edge_net(cover)                                        # (1,1,H,W)
        edge_mask = encoder.edge_epsilon + (1.0 - encoder.edge_epsilon) * edge_map # mask with ε floor
        sobel     = sobel_edges(cover)                                             # (1,1,H,W)

    print("Tensor stats:")
    _stats(edge_map,  "edge_map (learned)")
    _stats(edge_mask, "edge_mask (w/ floor)")
    _stats(sobel,     "sobel (classical)")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    to_pil_rgb(cover).save(          f"{args.out}/{stem}_cover.png")
    to_pil_gray(edge_map).save(      f"{args.out}/{stem}_edge_learned.png")
    to_pil_gray(edge_mask).save(     f"{args.out}/{stem}_edge_mask.png")
    to_pil_gray(sobel).save(         f"{args.out}/{stem}_edge_sobel.png")
    overlay(cover, edge_map).save(   f"{args.out}/{stem}_overlay.png")

    print(f"\nSaved to {args.out}/")
    print(f"  {stem}_cover.png         — original")
    print(f"  {stem}_edge_learned.png  — EdgeNet output (compare with sobel)")
    print(f"  {stem}_edge_mask.png     — actual perturbation mask (ε={encoder.edge_epsilon})")
    print(f"  {stem}_edge_sobel.png    — classical Sobel reference")
    print(f"  {stem}_overlay.png       — learned edges in red over cover")


if __name__ == "__main__":
    main()
