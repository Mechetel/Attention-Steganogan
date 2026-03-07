# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class BasicDecoder(nn.Module):
    """
    The BasicDecoder module takes a steganographic image and attempts to decode
    the embedded data tensor.
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def __init__(self, data_depth: int) -> None:
        super(BasicDecoder, self).__init__()
        self.version: str = '1'
        self.data_depth: int = data_depth
        self._build_layers()

    def _build_layers(self) -> None:
        self.layer1_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer1_bn   = nn.BatchNorm2d(32)
        self.layer2_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_bn   = nn.BatchNorm2d(32)
        self.layer3_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer3_bn   = nn.BatchNorm2d(32)
        self.output_conv = nn.Conv2d(32, self.data_depth, kernel_size=3, padding=1)

    def upgrade_legacy(self) -> None:
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.layer1_bn(self.layer1_conv(x)), inplace=True)
        x = F.leaky_relu(self.layer2_bn(self.layer2_conv(x)), inplace=True)
        x = F.leaky_relu(self.layer3_bn(self.layer3_conv(x)), inplace=True)
        return self.output_conv(x)


class DenseDecoder(nn.Module):
    """
    The DenseDecoder module takes a steganographic image and attempts to decode
    the embedded data tensor with dense connections.
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def __init__(self, data_depth: int) -> None:
        super(DenseDecoder, self).__init__()
        self.version: str = '1'
        self.data_depth: int = data_depth
        self._build_layers()

    def _build_layers(self) -> None:
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(96, self.data_depth, kernel_size=3, padding=1)

    def upgrade_legacy(self) -> None:
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(torch.cat([x1, x2], dim=1))), inplace=True)
        return self.conv4(torch.cat([x1, x2, x3], dim=1))
