#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Continue training a pretrained EdgeAwareDenseASPP SteganoGAN model with an
optional frozen steganalyzer attached as a *second critic* in the encoder
loss (adversarial training against a real cover/stego classifier).

Encoder loss with the steganalyzer term enabled becomes:

    L = L_iter (decoder + image MSE + WGAN critic)
      + λ_edge  · SobelEdgeRegularisation(edge_map, cover)
      + λ_vgg   · VGGPerceptualLoss(cover, S_final)
      + λ_stega · CrossEntropy(Steganalyzer(S_final → [0,1]), label = COVER)

Workflow
--------
  1. Load an existing EdgeAwareDenseASPP SteganoGAN checkpoint
     (e.g. ``models/div2k/edge_aspp_edge_aware_dense/<run-id>/weights.steg``).
  2. Optionally build a frozen pretrained steganalyzer
     (e.g. ``steganalyzers/runs/efficientnetsteg/.../epoch0041.pt``).
     Selected via the ``steganalyzer.network`` config key — same registry as
     ``steganalyzers/train.py``.
  3. Attach it to the trainer (`trainer.attach_steganalyzer(...)`).
  4. Continue training for ``epochs`` more epochs.
  5. Save the new run under
     ``models/div2k/edge_aspp_edge_aware_dense_efficientnetsteg/<timestamp>-dN/``.

Toggle the steganalyzer term off by setting ``CONFIG["steganalyzer"]["enabled"] = False``
or by leaving it ``False`` (the default) — the run then becomes plain
continuation training, useful as a no-adversary baseline.
"""

import gc
import json
import os
import sys
from time import time
from typing import Any, Dict

sys.warnoptions.append("ignore::UserWarning:multiprocessing.resource_tracker")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from steganogan import SteganoGAN
from steganogan.data import DataLoaderFactory

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from steganalyzers.models import (
    XuNet, YeNet, SRNet, YedroudjNet, ZhuNet, SIAStegNet, EfficientNetSteg,
)


# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {
    # ── Hardware ──────────────────────────────────────────────────────────────
    "gpu":                True,

    # ── Source model (the edge-aspp SteganoGAN checkpoint to continue from) ──
    # d1: 1774269623-d1/weights.steg
    # d2: 1774253954-d2/32.rsbpp-1.891735.p
    # d3: 1774256915-d3/32.rsbpp-2.405477.p
    # d4: 1774272817-d4/32.rsbpp-1.787670.p
    "source_checkpoint":  "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/weights.steg",
    "data_depth":         1,    # must match the source run's data_depth (d1=1, …, d4=4)

    # ── Output dir (a timestamped run dir is created inside this root) ───────
    "output_root":        "models/div2k/edge_aspp_edge_aware_dense_efficientnetsteg",

    # ── Continuation training ────────────────────────────────────────────────
    "epochs":             32,
    "batch_size":         8,
    "num_workers":        8,
    "crop_size":          360,

    # ── Data ─────────────────────────────────────────────────────────────────
    "dataset":            "div2k",
    # "data_root":          os.path.expanduser("~/Projects/datasets"),
    "data_root":          os.path.expanduser("/workspace/Attention-Steganogan/data"),

    # ── Second critic: a frozen steganalyzer in the encoder loss ─────────────
    # Set `enabled` False (default) to run plain continuation training.
    "steganalyzer": {
        "enabled":          False,

        # Which network to instantiate. Choices:
        #   "xunet" | "yenet" | "srnet" | "yedroudjnet"
        #   | "zhunet" | "siastegnet" | "efficientnetsteg"
        "network":          "efficientnetsteg",

        # Path to the .pt checkpoint produced by steganalyzers/train.py
        # (bundle with key "model_state"). Must match `network`.
        "checkpoint":       "steganalyzers/runs/efficientnetsteg/efficientnetsteg_1780660488/epoch0041.pt",

        # Weight of the steganalyzer cross-entropy term in the encoder loss.
        "lambda":           1.0,

        # Network-build hyper-parameters (must match the checkpoint).
        # Only the relevant subset is read for each `network`.
        "srm_trainable":    False,
        "tlu_threshold":    3.0,
        "abs_layer":        True,
        "clamp_val":        3.0,
        "ca_reduction":     8,
        "dropout":          0.4,
        "freeze_backbone":  False,
    },
}


# ── Steganalyzer registry ─────────────────────────────────────────────────────

def _build_steganalyzer(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build a steganalyzer instance from the `steganalyzer` config block."""
    choice = cfg["network"].lower()
    common = dict(in_channels=3, num_classes=2)

    if choice == "xunet":
        return XuNet(**common)
    elif choice == "yenet":
        return YeNet(**common,
                     srm_trainable=cfg.get("srm_trainable", False),
                     tlu_threshold=cfg.get("tlu_threshold", 3.0))
    elif choice == "srnet":
        return SRNet(**common)
    elif choice == "yedroudjnet":
        return YedroudjNet(**common,
                           abs_layer=cfg.get("abs_layer", True),
                           clamp_val=cfg.get("clamp_val", 3.0))
    elif choice == "zhunet":
        return ZhuNet(**common,
                      srm_trainable=cfg.get("srm_trainable", False))
    elif choice == "siastegnet":
        return SIAStegNet(**common,
                          srm_trainable=cfg.get("srm_trainable", False))
    elif choice == "efficientnetsteg":
        return EfficientNetSteg(**common,
                                freeze_backbone=cfg.get("freeze_backbone", False),
                                dropout=cfg.get("dropout", 0.4))
    else:
        raise ValueError(f"Unknown steganalyzer network: {choice!r}")


def _load_steganalyzer(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Build a steganalyzer, load its checkpoint, freeze, and move to device."""
    model      = _build_steganalyzer(cfg)
    ckpt_path  = cfg["checkpoint"]
    bundle     = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    epoch = bundle.get("epoch", "?")
    print(f"  Steganalyzer ({cfg['network']}) ← {ckpt_path}  (epoch {epoch})")
    return model.to(device)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)

    print("=" * 60)
    print("SteganoGAN Retraining (edge_aspp + frozen steganalyzer)")
    print("=" * 60)
    for k, v in CONFIG.items():
        print(f"  {k:<22}: {v}")
    print()

    # ── Output dir ────────────────────────────────────────────────────────────
    d = CONFIG["data_depth"]
    log_dir = os.path.join(
        CONFIG["output_root"],
        f"{int(time())}-d{d}",
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # ── Load source SteganoGAN model ──────────────────────────────────────────
    src = CONFIG["source_checkpoint"]
    if not os.path.isabs(src):
        src = os.path.join(_HERE, src)
    print(f"Loading source model: {src}")
    model = SteganoGAN.load(
        src,
        gpu=CONFIG["gpu"],
        verbose=True,
        log_dir=log_dir,
    )
    if model.data_depth != d:
        print(
            f"Warning: source data_depth={model.data_depth} ≠ CONFIG.data_depth={d}; "
            f"using model.data_depth."
        )
        d = model.data_depth
    print(f"  data_depth: {d}")
    print(f"  device    : {model.device}")
    print(f"  encoder   : {type(model.encoder).__name__}")
    print(f"  decoder   : {type(model.decoder).__name__}")
    print(f"  critic    : {type(model.critic).__name__ if model.critic else None}\n")

    # ── Optionally attach frozen steganalyzer ─────────────────────────────────
    stega_cfg = CONFIG["steganalyzer"]
    if stega_cfg.get("enabled", False):
        steganalyzer = _load_steganalyzer(stega_cfg, model.device)
        model._trainer.attach_steganalyzer(
            steganalyzer,
            lambda_stega=stega_cfg.get("lambda", 1.0),
        )
        print(f"  λ_stega   : {stega_cfg.get('lambda', 1.0)}\n")
    else:
        print("Steganalyzer disabled — plain continuation training.\n")

    # ── Data ──────────────────────────────────────────────────────────────────
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
    print(f"Val  : {len(val_loader.dataset)} images")
    print(f"Logs → {log_dir}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model.fit(train_loader, val_loader, epochs=CONFIG["epochs"])

    # ── Save final ────────────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "weights.steg")
    model.save(final_path)

    print(f"\n{'=' * 60}")
    print("Retraining complete.")
    print(f"  Source     : {src}")
    print(f"  Checkpoint : {final_path}")
    print(f"  Metrics    : {os.path.join(log_dir, 'metrics.log')}")
    print("=" * 60)

    if model.fit_metrics:
        print("\nFinal epoch metrics:")
        for k, v in model.fit_metrics.items():
            print(f"  {k:<30}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
