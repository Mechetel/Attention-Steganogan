#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SteganoGAN training script.

Supports all encoder variants, including:
  EdgeGuidedDualStreamUNetEncoder  (Ji, Zhang, Lv – Applied Sciences 2025)

Paper-recommended settings to reproduce Div2K results:
  encoder    : 'edge_unet'    data_depth : 1
  batch_size : 2              epochs     : 100
  T=10, eta=1.0, gamma=0.8, alpha=1.0, image_size=360
"""

import gc
import json
import os
import sys
import warnings
from time import time
from typing import Any, Dict

# Suppress the benign macOS resource-tracker semaphore warning that
# appears when os._exit() terminates DataLoader workers abruptly.
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be",
    category=UserWarning,
)

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch

from steganogan import SteganoGAN
from steganogan.models import (
    BasicEncoder, ResidualEncoder, DenseEncoder,
    EdgeGuidedDualStreamUNetEncoder,
    BasicDecoder, DenseDecoder,
    BasicCritic,
)
from steganogan.data import DataLoaderFactory


# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {
    # Hardware
    "gpu":           True,         # True | False

    # Architecture
    "encoder":       "dense",       # basic | residual | dense | edge_unet
    "decoder":       "dense",       # basic | dense
    "critic":        False,          # True | False (if True, adds a critic network for adversarial training)

    # Training
    "data_depth":    1,             # bits per pixel
    "epochs":        1,
    "batch_size":    8,
    "num_workers":   8,
    "crop_size":     30,          # 400 for edge_unet, 360 for others

    # Data paths
    "dataset":       "div2k",
    "data_root":     os.path.expanduser("~/Projects/datasets"), # os.path.expanduser("~/Attention-Steganogan/data"),

    # EdgeGuidedDualStreamUNetEncoder hyper-parameters
    # "T":             10,            # ConvGRU iterations
    # "eta":           1.0,           # perturbation step size
    # "gamma":         0.8,           # iterative loss decay factor
    # "alpha":         1.0,           # image-quality loss weight
    # "sobel_alpha":   1.0,           # edge enhancement strength
    # "hidden_ch":     32,            # ConvGRU hidden channels
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
    else:
        raise ValueError(f"Unknown encoder: {choice!r}")


_DECODER_MAP = {"basic": BasicDecoder, "dense": DenseDecoder}


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
        decoder    = _DECODER_MAP[CONFIG["decoder"]],
        critic     = BasicCritic if CONFIG["critic"] else None,
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

    # Cleanly shut down DataLoader worker processes before hard-exiting.
    # This releases semaphores and avoids the resource_tracker warning on macOS.
    for loader in (train_loader, val_loader):
        if hasattr(loader, "_iterator") and loader._iterator is not None:
            loader._iterator._shutdown_workers()
    del train_loader, val_loader
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
