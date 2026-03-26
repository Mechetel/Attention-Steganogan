#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness evaluation comparing edge_aspp and edge_unet on Div2k 1-bit.

Tests decoder accuracy under three perturbations applied to stego images:
  1. JPEG compression (quality=80)
  2. Additive Gaussian noise (sigma=0.01)
  3. Multiplicative Gaussian noise (sigma=0.01)

Usage:
    /Users/dmitryhoma/anaconda3/envs/pytorch/bin/python3 robustness_eval.py
"""

import io
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Ensure steganogan package is importable ───────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from steganogan.utils.checkpoint import ModelCheckpoint

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = {
    "edge_aspp": os.path.join(
        REPO, "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/25.rsbpp-0.973312.p"
    ),
    "edge_unet": os.path.join(
        REPO, "models/div2k/edge_unet_dense/1773386504/32.rsbpp-0.969229.p"
    ),
    "dense": os.path.join(
        REPO, "models/div2k/dense_dense/1773340283/32.rsbpp-0.965443.p"
    ),
}

VAL_DIR    = os.path.expanduser("~/Projects/datasets/div2k/val/_")
CROP_SIZE  = 360
NUM_IMAGES = 10
DATA_DEPTH = 1
DEVICE     = torch.device("cpu")
SEED       = 42

JPEG_QUALITY = 80
NOISE_SIGMA  = 0.01


# ── SSIM / PSNR helpers ───────────────────────────────────────────────────────
def psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = F.mse_loss(reconstructed, original).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def ssim_single(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> float:
    from torch.nn.functional import conv2d
    C1, C2 = 0.01**2, 0.03**2
    channel = x.shape[0]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1).to(x.device)
    pad = window_size // 2
    mu1 = conv2d(x, kernel, padding=pad, groups=channel)
    mu2 = conv2d(y, kernel, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    s1  = conv2d(x * x, kernel, padding=pad, groups=channel) - mu1_sq
    s2  = conv2d(y * y, kernel, padding=pad, groups=channel) - mu2_sq
    s12 = conv2d(x * y, kernel, padding=pad, groups=channel) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*s12 + C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return ssim_map.mean().item()


# ── Perturbation functions ────────────────────────────────────────────────────
def jpeg_compress(t: torch.Tensor) -> torch.Tensor:
    pil = transforms.ToPILImage()(t.cpu().clamp(0, 1))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return transforms.ToTensor()(Image.open(buf).convert("RGB")).to(t.device)


def additive_noise(t: torch.Tensor) -> torch.Tensor:
    return (t + torch.randn_like(t) * NOISE_SIGMA).clamp(0, 1)


def multiplicative_noise(t: torch.Tensor) -> torch.Tensor:
    return (t * (1.0 + torch.randn_like(t) * NOISE_SIGMA)).clamp(0, 1)


PERTURBATIONS = {
    "JPEG(80)":   jpeg_compress,
    "Additive":   additive_noise,
    "Mult.":      multiplicative_noise,
}


# ── Per-model evaluation ──────────────────────────────────────────────────────
def run_model(model, img_files):
    results = {name: {"err_bits": 0, "total_bits": 0, "psnr_sum": 0.0, "ssim_sum": 0.0}
               for name in PERTURBATIONS}
    preprocess = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
    ])
    with torch.no_grad():
        for img_path in img_files:
            pil   = Image.open(img_path).convert("RGB")
            cover = preprocess(pil).unsqueeze(0).to(DEVICE)
            msg   = torch.randint(0, 2, (1, DATA_DEPTH, CROP_SIZE, CROP_SIZE),
                                  dtype=torch.float32).to(DEVICE)
            stego = model.encoder(cover, msg)
            for name, fn in PERTURBATIONS.items():
                stego_p = fn(stego.squeeze(0)).unsqueeze(0)
                logits  = model.decoder(stego_p)
                decoded = (logits >= 0).float()
                results[name]["err_bits"]   += (decoded != msg).sum().item()
                results[name]["total_bits"] += msg.numel()
                results[name]["psnr_sum"]   += psnr(cover.squeeze(0), stego_p.squeeze(0))
                results[name]["ssim_sum"]   += ssim_single(cover.squeeze(0), stego_p.squeeze(0))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    img_files = sorted([
        os.path.join(VAL_DIR, f)
        for f in os.listdir(VAL_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:NUM_IMAGES]
    print(f"Using {len(img_files)} validation images\n")

    all_results = {}
    for model_name, ckpt_path in MODELS.items():
        print(f"Loading {model_name} …")
        model = ModelCheckpoint.load(ckpt_path, gpu=False, verbose=False)
        model.encoder.eval()
        model.decoder.eval()
        all_results[model_name] = run_model(model, img_files)
        print(f"  done.")

    n = len(img_files)
    pert_names = list(PERTURBATIONS.keys())
    metrics    = ["Error Rate (%)", "PSNR", "SSIM"]

    # Print comparison table
    print("\n" + "="*80)
    print("Robustness Comparison — Div2k 1-bit")
    print("="*80)
    col_w = 14
    header = f"{'Method':<12} {'Metric':<16}" + "".join(f"{p:>{col_w}}" for p in pert_names)
    print(header)
    print("-"*80)

    for model_name, res in all_results.items():
        errs  = [100 * res[p]["err_bits"] / res[p]["total_bits"] for p in pert_names]
        psnrs = [res[p]["psnr_sum"] / n                          for p in pert_names]
        ssims = [res[p]["ssim_sum"] / n                          for p in pert_names]
        print(f"{model_name:<12} {'Error Rate (%)':16}" + "".join(f"{v:>{col_w}.2f}" for v in errs))
        print(f"{'':12} {'PSNR':16}"                   + "".join(f"{v:>{col_w}.2f}" for v in psnrs))
        print(f"{'':12} {'SSIM':16}"                   + "".join(f"{v:>{col_w}.4f}" for v in ssims))
        print()

    # Markdown tables
    print("\n\n--- Markdown ---\n")
    for model_name, res in all_results.items():
        errs  = [100 * res[p]["err_bits"] / res[p]["total_bits"] for p in pert_names]
        psnrs = [res[p]["psnr_sum"] / n                          for p in pert_names]
        ssims = [res[p]["ssim_sum"] / n                          for p in pert_names]
        print(f"**{model_name}**")
        print(f"| Metric | {' | '.join(pert_names)} |")
        print("| --- | " + " | ".join([":---:"] * len(pert_names)) + " |")
        print(f"| Error Rate (%) | {' | '.join(f'{v:.2f}' for v in errs)} |")
        print(f"| PSNR | {' | '.join(f'{v:.2f}' for v in psnrs)} |")
        print(f"| SSIM | {' | '.join(f'{v:.4f}' for v in ssims)} |")
        print()


if __name__ == "__main__":
    main()
