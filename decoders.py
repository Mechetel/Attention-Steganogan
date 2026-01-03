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

    def __init__(self, data_depth):
        super(BasicDecoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        # Decoder layers
        self.layer1_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer1_bn = nn.BatchNorm2d(32)

        self.layer2_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_bn = nn.BatchNorm2d(32)

        self.layer3_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer3_bn = nn.BatchNorm2d(32)

        self.output_conv = nn.Conv2d(32, self.data_depth, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, x):
        # First layer
        x = self.layer1_conv(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.layer1_bn(x)

        # Second layer
        x = self.layer2_conv(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.layer2_bn(x)

        # Third layer
        x = self.layer3_conv(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.layer3_bn(x)

        # Output layer
        x = self.output_conv(x)

        return x


class DenseDecoder(nn.Module):
    """
    The DenseDecoder module takes a steganographic image and attempts to decode
    the embedded data tensor with dense connections.
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def __init__(self, data_depth):
        super(DenseDecoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        # First convolution block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolution block
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Third convolution block (dense connection)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Output convolution
        self.conv4 = nn.Conv2d(96, self.data_depth, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, x):
        # First block
        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1, inplace=True)
        x1 = self.bn1(x1)

        # Second block
        x2 = self.conv2(x1)
        x2 = F.leaky_relu(x2, inplace=True)
        x2 = self.bn2(x2)

        # Third block with dense connection
        x3_input = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x3_input)
        x3 = F.leaky_relu(x3, inplace=True)
        x3 = self.bn3(x3)

        # Output block with all dense connections
        x4_input = torch.cat([x1, x2, x3], dim=1)
        output = self.conv4(x4_input)

        return output
