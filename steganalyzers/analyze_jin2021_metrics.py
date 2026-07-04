#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection accuracy P_A and MMD (Jin et al., 2021) from an existing ROC results.json.

Reads the fpr/tpr curves already computed by analyze_steganogan_roc.py
(runs_steganogan_v3/steganogan_roc/results.json) and derives the two
metrics from the reference paper without re-running any model inference:

  detection_accuracy_pa : P_A = 1 - P_E, threshold-optimal detection
                           accuracy. Exact — derived losslessly from the
                           stored fpr/tpr curve.
  mmd                    : RBF-kernel Maximum Mean Discrepancy between the
                           cover- and stego-score distributions. The stored
                           results only keep the aggregated fpr/tpr curve,
                           not the raw per-image scores, so the score of
                           each sample is reconstructed as its rank
                           quantile in the curve (same order as the
                           original scores) — a proxy for the true MMD,
                           monotonic-order-equivalent but not on the
                           original probability scale.

Usage
-----
    python steganalyzers/analyze_jin2021_metrics.py
"""

import json
import os
import sys
from typing import Dict, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from steganalyzers.training import SteganalysisMetrics

RESULTS_JSON = os.path.join(_HERE, "runs_steganogan_v3", "steganogan_roc", "results.json")
OUTPUT_DIR   = os.path.join(_HERE, "runs_steganogan_v3", "steganogan_roc")
OUTPUT_JSON  = os.path.join(OUTPUT_DIR, "jin2021_metrics.json")
OUTPUT_MD    = os.path.join(OUTPUT_DIR, "jin2021_metrics_table.md")
OUTPUT_PLOT  = os.path.join(OUTPUT_DIR, "jin2021_metrics.png")


# ── Reconstruction ──────────────────────────────────────────────────────────────

def _reconstruct_labels_and_scores(fpr: list, tpr: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover a (pseudo_scores, labels) pair, in descending-score order, from
    a cumulative fpr/tpr curve produced by SteganalysisMetrics.auc_roc-style
    thresholding (padded with a (0,0) start point and a (1,1) end point).

    At each real sample step exactly one of fpr/tpr increases: tpr
    increasing marks a stego sample, fpr increasing marks a cover sample.
    """
    fpr_arr = np.asarray(fpr, dtype=np.float64)
    tpr_arr = np.asarray(tpr, dtype=np.float64)

    d_fpr = np.diff(fpr_arr)[:-1]   # drop the redundant trailing (1,1)-pad diff
    d_tpr = np.diff(tpr_arr)[:-1]

    labels = (d_tpr > d_fpr).astype(int)   # 1 = stego step, 0 = cover step
    n = len(labels)
    # Descending quantile in (0, 1), matching the original sort order.
    pseudo_scores = 1.0 - (np.arange(n) + 0.5) / n
    return pseudo_scores, labels


def compute_jin2021_metrics(fpr: list, tpr: list) -> Dict[str, float]:
    scores, labels = _reconstruct_labels_and_scores(fpr, tpr)
    return {
        "detection_accuracy_pa": SteganalysisMetrics.detection_accuracy_pa(scores, labels),
        "mmd":                   SteganalysisMetrics.mmd_rbf(scores, labels),
    }


# ── Table ─────────────────────────────────────────────────────────────────────

def build_markdown_tables(results: Dict[str, Dict[str, Dict]]) -> str:
    networks = list(results.keys())
    variants = list(next(iter(results.values())).keys())

    lines = [
        "# Jin et al. (2021) Metrics — Detection Accuracy P_A and MMD",
        "",
        "Derived from the ROC curves in `results.json` (no re-inference needed).",
        "P_A is exact; MMD is a rank-based proxy (see script docstring).",
        "",
    ]

    for metric_key, metric_title in [
        ("detection_accuracy_pa", "Detection Accuracy (P_A = 1 - P_E)"),
        ("mmd", "MMD (RBF kernel, rank-based proxy)"),
    ]:
        lines.append(f"## {metric_title}")
        lines.append("")
        lines.append("| Detector | " + " | ".join(networks) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(networks)) + " |")
        for variant in variants:
            row = [variant]
            for net in networks:
                row.append(f"{results[net][variant][metric_key]:.6f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_grouped_bars(results: Dict[str, Dict[str, Dict]], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    networks = list(results.keys())
    variants = list(next(iter(results.values())).keys())
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, (metric_key, metric_title) in zip(
        axes,
        [
            ("detection_accuracy_pa", "Detection Accuracy P_A"),
            ("mmd", "MMD (rank-based proxy)"),
        ],
    ):
        x = np.arange(len(networks))
        width = 0.8 / len(variants)
        for i, variant in enumerate(variants):
            values = [results[net][variant][metric_key] for net in networks]
            ax.bar(x + i * width, values, width, label=variant,
                   color=colors[i % len(colors)])

        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels(networks, rotation=20, ha="right")
        ax.set_ylabel(metric_title)
        ax.set_title(metric_title)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading {RESULTS_JSON} ...")
    with open(RESULTS_JSON) as f:
        raw_results = json.load(f)

    jin_results: Dict[str, Dict[str, Dict]] = {}
    for network, variants in raw_results.items():
        jin_results[network] = {}
        for variant, m in variants.items():
            metrics = compute_jin2021_metrics(m["fpr"], m["tpr"])
            jin_results[network][variant] = metrics
            print(f"  {network:<18} {variant:<18} "
                  f"P_A={metrics['detection_accuracy_pa']:.4f}  "
                  f"MMD={metrics['mmd']:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(jin_results, f, indent=2)
    print(f"\nMetrics saved -> {OUTPUT_JSON}")

    with open(OUTPUT_MD, "w") as f:
        f.write(build_markdown_tables(jin_results))
    print(f"Tables saved  -> {OUTPUT_MD}")

    plot_grouped_bars(jin_results, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
