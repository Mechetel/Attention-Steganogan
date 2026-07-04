# -*- coding: utf-8 -*-
"""
Steganalysis evaluation metrics.

All methods are static; the class is a pure-utility namespace.

Metrics
-------
accuracy          : fraction of correctly classified images
balanced_accuracy : mean per-class recall (robust to imbalanced test sets)
auc_roc           : Area Under the ROC Curve
tpr_at_fpr        : True Positive Rate at a fixed False Positive Rate
              (standard ALASKA2 evaluation metric at FPR = 0.1)
precision         : TP / (TP + FP)
recall            : TP / (TP + FN)
f1                : harmonic mean of precision and recall
detection_accuracy_pa : P_A = 1 - P_E, threshold-optimal detection accuracy,
              where P_E = min over decision threshold of (P_FA + P_MD) / 2
              (Jin et al., "Feature Selection of the Rich Model Based on the
              Correlation of Feature Components", Security and Communication
              Networks, 2021)
mmd               : Maximum Mean Discrepancy (RBF kernel) between the
              cover- and stego-score distributions — a classifier-threshold
              independent separability measure (Jin et al., 2021)
"""

from typing import Dict, Optional

import torch
import numpy as np


class SteganalysisMetrics:
    """Static utility namespace for steganalysis performance metrics."""

    @staticmethod
    def from_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute the full metric report from model logits and ground-truth labels.

        Parameters
        ----------
        logits : (N, 2) raw model outputs (before softmax)
        labels : (N,)   integer labels {0=cover, 1=stego}

        Returns
        -------
        dict with keys: accuracy, balanced_accuracy, auc_roc,
                        tpr_at_fpr10, precision, recall, f1,
                        detection_accuracy_pa, mmd
        """
        probs  = torch.softmax(logits.float(), dim=1)[:, 1]   # P(stego)
        preds  = logits.argmax(dim=1)
        labels = labels.long()

        probs_np  = probs.detach().cpu().numpy()
        preds_np  = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        acc      = SteganalysisMetrics.accuracy(preds_np, labels_np)
        bal_acc  = SteganalysisMetrics.balanced_accuracy(preds_np, labels_np)
        auc      = SteganalysisMetrics.auc_roc(probs_np, labels_np)
        tpr10    = SteganalysisMetrics.tpr_at_fpr(probs_np, labels_np, fpr_target=0.1)
        prec, rec, f1 = SteganalysisMetrics.precision_recall_f1(preds_np, labels_np)
        pa       = SteganalysisMetrics.detection_accuracy_pa(probs_np, labels_np)
        mmd      = SteganalysisMetrics.mmd_rbf(probs_np, labels_np)

        return {
            "accuracy":              acc,
            "balanced_accuracy":     bal_acc,
            "auc_roc":               auc,
            "tpr_at_fpr10":          tpr10,
            "precision":             prec,
            "recall":                rec,
            "f1":                    f1,
            "detection_accuracy_pa": pa,
            "mmd":                   mmd,
        }

    # ── Individual metrics ─────────────────────────────────────────────────────

    @staticmethod
    def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
        return float((preds == labels).mean())

    @staticmethod
    def balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
        """Mean per-class recall — handles class imbalance."""
        classes = np.unique(labels)
        recalls = []
        for c in classes:
            mask = labels == c
            if mask.sum() == 0:
                continue
            recalls.append((preds[mask] == c).mean())
        return float(np.mean(recalls)) if recalls else 0.0

    @staticmethod
    def auc_roc(
        probs:  np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Area Under the ROC Curve (trapezoidal rule)."""
        # Sort by descending score
        order = np.argsort(-probs)
        sorted_labels = labels[order]

        n_pos = sorted_labels.sum()
        n_neg = len(sorted_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_vals, fpr_vals = [0.0], [0.0]
        tp = fp = 0
        for lbl in sorted_labels:
            if lbl == 1:
                tp += 1
            else:
                fp += 1
            tpr_vals.append(tp / n_pos)
            fpr_vals.append(fp / n_neg)
        tpr_vals.append(1.0)
        fpr_vals.append(1.0)

        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(trapz(tpr_vals, fpr_vals))

    @staticmethod
    def tpr_at_fpr(
        probs:      np.ndarray,
        labels:     np.ndarray,
        fpr_target: float = 0.1,
    ) -> float:
        """
        True Positive Rate at a fixed False Positive Rate.

        This is the primary ALASKA2 challenge metric evaluated at FPR = 0.1.
        """
        order = np.argsort(-probs)
        sorted_labels = labels[order]

        n_pos = sorted_labels.sum()
        n_neg = len(sorted_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0

        tp = fp = 0
        for lbl in sorted_labels:
            if lbl == 1:
                tp += 1
            else:
                fp += 1
                if fp / n_neg >= fpr_target:
                    return float(tp / n_pos)
        return float(tp / n_pos)

    @staticmethod
    def precision_recall_f1(
        preds:  np.ndarray,
        labels: np.ndarray,
        pos_class: int = 1,
    ):
        """Precision, recall, and F1 for the positive (stego) class."""
        tp = int(((preds == pos_class) & (labels == pos_class)).sum())
        fp = int(((preds == pos_class) & (labels != pos_class)).sum())
        fn = int(((preds != pos_class) & (labels == pos_class)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return float(precision), float(recall), float(f1)

    @staticmethod
    def detection_accuracy_pa(
        probs:  np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Threshold-optimal detection accuracy P_A = 1 - P_E (Jin et al., 2021).

        P_E = min over decision threshold of (P_FA + P_MD) / 2, where P_FA is
        the false-alarm rate (cover misclassified as stego) and P_MD is the
        missed-detection rate (stego misclassified as cover). Unlike plain
        `accuracy` (fixed 0.5 softmax threshold), this sweeps every achievable
        operating point on the ROC curve and reports the best one — matching
        how the reference paper reports ensemble-classifier detection accuracy.
        """
        order = np.argsort(-probs)
        sorted_labels = labels[order]

        n_pos = sorted_labels.sum()
        n_neg = len(sorted_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp = fp = 0
        min_pe = 1.0
        for lbl in sorted_labels:
            if lbl == 1:
                tp += 1
            else:
                fp += 1
            p_fa = fp / n_neg
            p_md = 1.0 - (tp / n_pos)
            pe = 0.5 * (p_fa + p_md)
            if pe < min_pe:
                min_pe = pe
        return float(1.0 - min_pe)

    @staticmethod
    def mmd_rbf(
        probs:       np.ndarray,
        labels:      np.ndarray,
        max_samples: int = 2000,
        seed:        int = 0,
    ) -> float:
        """
        Maximum Mean Discrepancy (RBF kernel) between cover- and stego-score
        distributions (Jin et al., 2021).

        The reference paper computes MMD between cover/stego Rich Model
        feature distributions as a classifier-independent separability
        measure; here the classifier's P(stego) score stands in for the
        feature vector. Kernel bandwidth is set via the median heuristic on
        the pooled scores. Larger MMD ⇒ more separable cover/stego
        distributions. Groups are subsampled to `max_samples` each to keep
        the O(N^2) kernel computation bounded.
        """
        cover = probs[labels == 0]
        stego = probs[labels == 1]
        if len(cover) == 0 or len(stego) == 0:
            return 0.0

        rng = np.random.default_rng(seed)
        if len(cover) > max_samples:
            cover = rng.choice(cover, size=max_samples, replace=False)
        if len(stego) > max_samples:
            stego = rng.choice(stego, size=max_samples, replace=False)

        pooled = np.concatenate([cover, stego])
        diffs = np.abs(pooled[:, None] - pooled[None, :])
        nonzero = diffs[diffs > 0]
        median_dist = np.median(nonzero) if nonzero.size > 0 else 1.0
        gamma = 1.0 / (2.0 * median_dist ** 2 + 1e-12)

        def _rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.exp(-gamma * (a[:, None] - b[None, :]) ** 2)

        k_cc = _rbf(cover, cover).mean()
        k_ss = _rbf(stego, stego).mean()
        k_cs = _rbf(cover, stego).mean()

        mmd_sq = k_cc - 2.0 * k_cs + k_ss
        return float(np.sqrt(max(mmd_sq, 0.0)))

    # ── Aggregation helpers ────────────────────────────────────────────────────

    @staticmethod
    def average(metric_lists: Dict[str, list]) -> Dict[str, float]:
        """Average a dict of metric lists (one entry per batch)."""
        return {k: float(np.mean(v)) for k, v in metric_lists.items()}
