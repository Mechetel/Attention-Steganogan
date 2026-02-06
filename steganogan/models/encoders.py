# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class BasicEncoder(nn.Module):
    """
    The BasicEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """
    add_image = False

    def __init__(self, data_depth):
        super(BasicEncoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        # Feature extraction layers
        self.feature_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.feature_bn = nn.BatchNorm2d(32)

        # Processing layers
        self.layer1_conv = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.layer1_bn = nn.BatchNorm2d(32)

        self.layer2_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_bn = nn.BatchNorm2d(32)

        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, image, data):
        # Feature extraction
        x = self.feature_conv(image)
        x = F.leaky_relu(x, inplace=True)
        x = self.feature_bn(x)

        # Concatenate with data and process
        x = torch.cat([x, data], dim=1)
        x = self.layer1_conv(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.layer1_bn(x)

        x = self.layer2_conv(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.layer2_bn(x)

        x = self.output_conv(x)
        x = torch.tanh(x)

        return x


class ResidualEncoder(nn.Module):
    """
    The ResidualEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    def __init__(self, data_depth):
        super(ResidualEncoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        # Feature extraction layers
        self.feature_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.feature_bn = nn.BatchNorm2d(32)

        # Processing layers
        self.layer1_conv = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.layer1_bn = nn.BatchNorm2d(32)

        self.layer2_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_bn = nn.BatchNorm2d(32)

        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, image, data):
        # Feature extraction
        x = self.feature_conv(image)
        x = F.leaky_relu(x, inplace=True)
        x = self.feature_bn(x)

        # Concatenate with data and process
        x = torch.cat([x, data], dim=1)
        x = self.layer1_conv(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.layer1_bn(x)

        x = self.layer2_conv(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.layer2_bn(x)

        x = self.output_conv(x)

        # Add residual connection (original image)
        x = image + x

        return x


class DenseEncoder(nn.Module):
    """
    The DenseEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image with dense connections.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    def __init__(self, data_depth):
        super(DenseEncoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        # First convolution block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolution block
        self.conv2 = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Third convolution block (dense connection)
        self.conv3 = nn.Conv2d(64 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Output convolution
        self.conv4 = nn.Conv2d(96 + self.data_depth, 3, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, image, data):
        # First block
        x1 = self.conv1(image)
        x1 = F.leaky_relu(x1, inplace=True)
        x1 = self.bn1(x1)

        # Second block with dense connection
        x2_input = torch.cat([x1, data], dim=1)
        x2 = self.conv2(x2_input)
        x2 = F.leaky_relu(x2, inplace=True)
        x2 = self.bn2(x2)

        # Third block with dense connections
        x3_input = torch.cat([x1, x2, data], dim=1)
        x3 = self.conv3(x3_input)
        x3 = F.leaky_relu(x3, inplace=True)
        x3 = self.bn3(x3)

        # Output block with all dense connections
        x4_input = torch.cat([x1, x2, x3, data], dim=1)
        output = self.conv4(x4_input)

        # Add residual connection (original image)
        output = image + output

        return output
