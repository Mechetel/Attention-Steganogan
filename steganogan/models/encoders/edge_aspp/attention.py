# -*- coding: utf-8 -*-
"""
Attention modules for EdgeAwareDenseASPP ablation experiments.

  MSMAModule  — re-exported from edge_unet (Multi-Scale Median Attention)
  ECAModule   — Efficient Channel Attention (Wang et al., ECA-Net, CVPR 2020)
  CBAMModule  — Convolutional Block Attention Module (Woo et al., ECCV 2018)

Factory
-------
  build_attention(channels, attention_type) → nn.Module | None
    attention_type: 'msma' | 'eca' | 'cbam' | '' (none)
"""

import math

import torch
import torch.nn as nn

from ..edge_unet.attention import MSMAModule


# ── ECA ───────────────────────────────────────────────────────────────────────

class ECAModule(nn.Module):
    """
    Efficient Channel Attention (Wang et al., ECA-Net, CVPR 2020).

    Global average pooling → 1-D conv (adaptive kernel) → sigmoid gating.

    Input / Output shape: (N, C, H, W)
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d   = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        w = self.conv1d(w).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(w)


# ── CBAM ──────────────────────────────────────────────────────────────────────

class _CBAMChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        avg = x.view(N, C, -1).mean(-1)
        mx  = x.view(N, C, -1).max(-1).values
        return x * torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(N, C, 1, 1)


class _CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAMModule(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Sequentially applies channel attention then spatial attention.

    Input / Output shape: (N, C, H, W)
    """

    def __init__(self, channels: int, reduction: int = 4,
                 spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = _CBAMChannelAttention(channels, reduction)
        self.spatial = _CBAMSpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ── Factory ───────────────────────────────────────────────────────────────────

def build_attention(channels: int, attention_type: str) -> nn.Module | None:
    """Return the requested attention module or None for empty string."""
    if   attention_type == "msma": return MSMAModule(channels)
    elif attention_type == "eca":  return ECAModule(channels)
    elif attention_type == "cbam": return CBAMModule(channels)
    return None
