#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SteganoGAN training script.

Supports all encoder variants, including:
  EdgeGuidedDualStreamUNetEncoder  (Ji, Zhang, Lv – Applied Sciences 2025)
  EdgeAwareDenseASPPEncoder        (Novel: DenseASPP + MSMA + learned edge detection)
  DepthAgnosticEncoder             (Novel: D never baked in weights — one model for any depth)

Paper-recommended settings to reproduce Div2K results:
  encoder    : 'edge_unet'    data_depth : 1
  batch_size : 2              epochs     : 100
  T=10, eta=1.0, gamma=0.8, alpha=100.0, image_size=360

Edge-aware DenseASPP settings:
  encoder    : 'edge_aspp'    data_depth : 1
  decoder    : 'edge_aware_dense'
  critic     : 'multiscale_edge'
  T=8, eta=1.0, gamma=0.8, alpha=100.0, hidden_ch=48

Depth-agnostic settings (train once, run at any D=1..data_depth):
  encoder    : 'agnostic'     data_depth : 6
  decoder    : 'agnostic'
  critic     : False | 'basic' | 'multiscale_edge'
"""

import gc
import json
import multiprocessing
import os
import sys

# The resource_tracker subprocess (which prints the semaphore leak warning)
# is spawned with Python's -W flags copied from sys.warnoptions.  Adding the
# filter here — before any DataLoader worker is created — causes the filter
# to be forwarded to the subprocess, silencing the warning at its source.
sys.warnoptions.append("ignore::UserWarning:multiprocessing.resource_tracker")

from time import time
from typing import Any, Dict

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from steganogan import SteganoGAN
from steganogan.models import (
    BasicEncoder, ResidualEncoder, DenseEncoder,
    EdgeGuidedDualStreamUNetEncoder,
    DepthAgnosticEncoder,
    BasicDecoder, DenseDecoder,
    DepthAgnosticDecoder,
    BasicCritic,
)
from steganogan.models.encoders.edge_aspp import EdgeAwareDenseASPPEncoder
from steganogan.models.decoders.decoders import EdgeAwareDenseDecoder
from steganogan.models.critics.critics import MultiScaleEdgeAwareCritic
from steganogan.data import DataLoaderFactory


# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {
    # Hardware
    "gpu":           True,         # True | False

    # Architecture
    "encoder":       "edge_aspp",        # basic | residual | dense | edge_unet | edge_aspp | agnostic
    "decoder":       "edge_aware_dense", # basic | dense | edge_aware_dense | agnostic
    "critic":        "multiscale_edge",  # basic | multiscale_edge | False

    # Training
    "data_depth":    1,             # bits per pixel
    "epochs":        32,
    "batch_size":    8,
    "num_workers":   8,
    "crop_size":     360,

    # Data paths
    "dataset":       "div2k",
    # "data_root":     os.path.expanduser("~/Projects/datasets"),
    "data_root":     os.path.expanduser("/workspace/Attention-Steganogan/data"),

    # EdgeGuidedDualStreamUNetEncoder hyper-parameters
    # "T":             10,            # ConvGRU iterations
    # "eta":           1.0,           # perturbation step size
    # "gamma":         0.8,           # iterative loss decay factor
    # "alpha":         100.0,         # image-quality loss weight
    # "sobel_alpha":   1.0,           # edge enhancement strength
    # "hidden_ch":     32,            # ConvGRU hidden channels

    #        (T,  alpha, lambda_edge, lambda_vgg, hidden_ch) tuples to try for EdgeAwareDenseASPPEncoder:
    # conf_1 (4,  100.0, 0.01,        0.1,        48) +
    # conf_2 (8,  100.0, 0.01,        0.5,        48) +
    # conf_3 (4,  10.0,  0.01,        0.1,        24) +
    # conf_4 (4,  100.0, 0.5,         0.1,        36) +
    # conf_5 (4,  1.0,   0.01,        1.0,        48) +
    # conf_6 (10, 100.0, 0.01,        0.5,        24) +
    # EdgeAwareDenseASPPEncoder hyper-parameters
    "T":             4,             # ConvGRU iterations (fewer needed with DenseASPP)
    "eta":           1.0,           # perturbation step size
    "gamma":         0.8,           # iterative loss decay factor
    "alpha":         100.0,         # image-quality loss weight
    "hidden_ch":     48,            # ConvGRU hidden channels
    "edge_epsilon":  0.05,          # edge mask floor (prevents dead gradients)
    "lambda_edge":   0.01,          # edge regularisation weight
    "lambda_vgg":    0.1,           # VGG perceptual loss weight
}

# ── Encoder / decoder / critic registries ─────────────────────────────────────

def _build_encoder(cfg: Dict[str, Any]) -> torch.nn.Module:
    d = cfg["data_depth"]
    choice = cfg["encoder"]
    if   choice == "basic":     return BasicEncoder(d)
    elif choice == "residual":  return ResidualEncoder(d)
    elif choice == "dense":     return DenseEncoder(d)
    elif choice == "edge_unet":
        return EdgeGuidedDualStreamUNetEncoder(
            data_depth  = d,
            T           = cfg["T"],
            eta         = cfg["eta"],
            gamma       = cfg["gamma"],
            alpha       = cfg["alpha"],
            sobel_alpha = cfg["sobel_alpha"],
            hidden_ch   = cfg["hidden_ch"],
        )
    elif choice == "edge_aspp":
        enc = EdgeAwareDenseASPPEncoder(
            data_depth   = d,
            T            = cfg.get("T", 8),
            eta          = cfg.get("eta", 1.0),
            gamma        = cfg.get("gamma", 0.8),
            alpha        = cfg.get("alpha", 100.0),
            hidden_ch    = cfg.get("hidden_ch", 48),
            edge_epsilon = cfg.get("edge_epsilon", 0.05),
        )
        # Store loss weights on encoder for Trainer to pick up
        enc.lambda_edge = cfg.get("lambda_edge", 0.01)
        enc.lambda_vgg  = cfg.get("lambda_vgg", 0.1)
        return enc
    elif choice == "agnostic":
        return DepthAgnosticEncoder(d)
    else:
        raise ValueError(f"Unknown encoder: {choice!r}")


def _build_decoder(cfg: Dict[str, Any]) -> type:
    choice = cfg["decoder"]
    if   choice == "basic":            return BasicDecoder
    elif choice == "dense":            return DenseDecoder
    elif choice == "edge_aware_dense": return EdgeAwareDenseDecoder
    elif choice == "agnostic":         return DepthAgnosticDecoder
    else:
        raise ValueError(f"Unknown decoder: {choice!r}")


def _build_critic(cfg: Dict[str, Any]) -> type | None:
    choice = cfg["critic"]
    if   choice is False or choice is None: return None
    elif choice is True:                    return BasicCritic
    elif choice == "basic":                 return BasicCritic
    elif choice == "multiscale_edge":       return MultiScaleEdgeAwareCritic
    else:
        raise ValueError(f"Unknown critic: {choice!r}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)

    print("=" * 60)
    print("SteganoGAN Training")
    print("=" * 60)
    for k, v in CONFIG.items():
        print(f"  {k:<18}: {v}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    train_root = os.path.join(CONFIG["data_root"], CONFIG["dataset"], "train")
    val_root   = os.path.join(CONFIG["data_root"], CONFIG["dataset"], "val")

    train_loader, val_loader = DataLoaderFactory.create(
        train_root  = train_root,
        val_root    = val_root,
        batch_size  = CONFIG["batch_size"],
        num_workers = CONFIG["num_workers"],
        crop_size   = CONFIG["crop_size"],
    )
    print(f"Train: {len(train_loader.dataset)} images")
    print(f"Val  : {len(val_loader.dataset)} images\n")

    # ── Model ─────────────────────────────────────────────────────────────
    log_dir = os.path.join(
        "models",
        f"{CONFIG['encoder']}_{CONFIG['decoder']}",
        str(int(time())),
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    model = SteganoGAN(
        data_depth = CONFIG["data_depth"],
        encoder    = _build_encoder(CONFIG),
        decoder    = _build_decoder(CONFIG),
        critic     = _build_critic(CONFIG),
        gpu        = CONFIG["gpu"],
        verbose    = True,
        log_dir    = log_dir,
    )
    print(model)
    print(f"\nLogs → {log_dir}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    model.fit(train_loader, val_loader, epochs=CONFIG["epochs"])

    # ── Save ──────────────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "weights.steg")
    model.save(final_path)

    print(f"\n{'=' * 60}")
    print("Training complete.")
    print(f"  Checkpoint : {final_path}")
    print(f"  Metrics    : {os.path.join(log_dir, 'metrics.log')}")
    print("=" * 60)

    if model.fit_metrics:
        print("\nFinal epoch metrics:")
        for k, v in model.fit_metrics.items():
            print(f"  {k:<30}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    os._exit(0)


if __name__ == "__main__":
    main()
