# -*- coding: utf-8 -*-
"""
WGAN critic (discriminator) for adversarial steganography training.

Reference: Arjovsky et al., "Wasserstein GAN", ICML 2017.
"""

import torch
import torch.nn as nn

from ..base import BaseCritic


class BasicCritic(BaseCritic):
    """
    Fully-convolutional WGAN critic.

    Produces a per-pixel score map; the caller averages it to get a scalar
    Wasserstein distance proxy (higher = more real/cover-like).

    Architecture: Conv3×3 → LeakyReLU → BN  (×2) → Conv3×3 → score map

    Input : image (N, 3, H, W)
    Output: score (N, 1, H, W)

    Parameters
    ----------
    data_depth : accepted for API compatibility; not used internally
    """

    def __init__(self, data_depth: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,  1, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)
