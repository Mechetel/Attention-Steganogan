# -*- coding: utf-8 -*-
"""
Perceptual image quality metrics: SSIM implementation and a torch-friendly
module-level helper function.
"""

from math import exp
from typing import Tuple

import torch
from torch.nn.functional import conv2d


class SSIMCalculator:
    """
    Structural Similarity Index Measure (SSIM) between two image tensors.

    All methods are static; this class is a namespace for SSIM utilities.
    Reference: Wang et al., "Image quality assessment: from error visibility
    to structural similarity", IEEE TIP 2004.
    """

    @staticmethod
    def gaussian_kernel(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        """
        Build a normalised 1-D Gaussian window.

        Parameters
        ----------
        window_size : number of taps
        sigma       : standard deviation
        """
        weights = torch.tensor(
            [exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
             for x in range(window_size)]
        )
        return weights / weights.sum()

    @staticmethod
    def build_window(window_size: int, n_channels: int) -> torch.Tensor:
        """
        Expand a 1-D Gaussian into a 2-D separable filter for *n_channels*.

        Returns a tensor of shape (n_channels, 1, window_size, window_size).
        """
        k1d = SSIMCalculator.gaussian_kernel(window_size).unsqueeze(1)
        k2d = k1d.mm(k1d.t()).float().unsqueeze(0).unsqueeze(0)
        return k2d.expand(n_channels, 1, window_size, window_size).contiguous()

    @staticmethod
    def _compute(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        n_channels: int,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Low-level SSIM computation (no window creation overhead)."""
        pad = window_size // 2

        mu1 = conv2d(img1, window, padding=pad, groups=n_channels)
        mu2 = conv2d(img2, window, padding=pad, groups=n_channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = conv2d(img1 * img1, window, padding=pad, groups=n_channels) - mu1_sq
        sigma2_sq = conv2d(img2 * img2, window, padding=pad, groups=n_channels) - mu2_sq
        sigma12   = conv2d(img1 * img2, window, padding=pad, groups=n_channels) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = (
            ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
            / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        )
        return ssim_map.mean() if reduce else ssim_map.mean(1).mean(1).mean(1)

    @staticmethod
    def calculate(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute SSIM between two batches of images.

        Parameters
        ----------
        img1, img2  : (N, C, H, W) image tensors in the same value range
        window_size : Gaussian kernel size (default 11)
        reduce      : if True, return a scalar mean; else per-image scores
        """
        n_channels = img1.size(1)
        window = SSIMCalculator.build_window(window_size, n_channels)
        window = window.to(img1.device).type_as(img1)
        return SSIMCalculator._compute(img1, img2, window, window_size, n_channels, reduce)


# ── Convenience alias ─────────────────────────────────────────────────────────

def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """Module-level SSIM helper (backward-compatible signature)."""
    return SSIMCalculator.calculate(img1, img2, window_size, reduce=size_average)


def first_element(storage: object, loc: object) -> object:
    """Pickle map_location helper – returns the first argument."""
    return storage
