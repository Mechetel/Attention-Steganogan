# -*- coding: utf-8 -*-
"""
Decoder architectures for SteganoGAN.

  BasicDecoder : 3-layer CNN
  DenseDecoder : DenseNet-style with dense skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseDecoder


class BasicDecoder(BaseDecoder):
    """
    3-layer CNN decoder.

    Three conv+BN+LeakyReLU layers, then a 1×1 output projection to D channels.

    Input : stego  (N, 3, H, W)
    Output: logits (N, D, H, W)  — apply threshold ≥0 for recovered bits
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.out   = nn.Conv2d(32, data_depth, kernel_size=3, padding=1)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(self.conv1(stego)), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), inplace=True)
        return self.out(x)


class DenseDecoder(BaseDecoder):
    """
    DenseNet-style decoder with dense skip connections.

    Each layer receives all previous feature maps concatenated, giving
    it direct access to low-level and high-level representations.

    Input : stego  (N, 3, H, W)
    Output: logits (N, D, H, W)
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.out   = nn.Conv2d(96, data_depth, kernel_size=3, padding=1)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        x1 = F.leaky_relu(self.bn1(self.conv1(stego)), inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(torch.cat([x1, x2], dim=1))), inplace=True)
        return self.out(torch.cat([x1, x2, x3], dim=1))
