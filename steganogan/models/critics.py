# -*- coding: utf-8 -*-
"""
SteganoGAN Critic (Discriminator)

The critic distinguishes cover images from stego images using a
Wasserstein GAN (WGAN) formulation. When enabled, adversarial training
pushes the encoder to produce stego images that are statistically
indistinguishable from real cover images.

References:
- Arjovsky et al., "Wasserstein GAN", ICML 2017
- SteganoGAN: DAI-Lab/SteganoGAN (original critic architecture)
"""

import torch
from torch import nn


class BasicCritic(nn.Module):
    """
    Basic Critic (Discriminator) for WGAN-based adversarial training.

    A fully-convolutional network that takes an image and produces a
    per-pixel score map. The caller takes torch.mean() of the output
    to get a scalar score (higher = more likely to be a real cover image).

    Architecture: 3 conv layers (3→32→32→1) with LeakyReLU + BN.

    Input: (N, 3, H, W)
    Output: (N, 1, H, W)
    """

    def __init__(self, data_depth: int) -> None:
        super(BasicCritic, self).__init__()
        self.version: str = '1'

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def upgrade_legacy(self) -> None:
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
