# -*- coding: utf-8 -*-
"""
Encoder architectures for SteganoGAN.

Includes:
  - BasicEncoder       : simple 3-layer CNN (SteganoGAN baseline)
  - ResidualEncoder    : BasicEncoder + residual connection
  - DenseEncoder       : DenseNet-style encoder (SteganoGAN default)
  - EdgeGuidedDualStreamUNetEncoder : full implementation of
      "Edge-Guided Dual-Stream U-Net for Secure Image Steganography"
      (Ji, Zhang, Lv – Applied Sciences 2025)

EdgeGuidedDualStreamUNetEncoder components
─────────────────────────────────────────
1. Sobel edge enhancement  (Section 3.1.1)
2. Dual-stream U-Net contracting path
      • original-image stream + MSMA attention  (Section 3.2)
      • edge-enhanced stream  + InceptionDMK    (Section 3.3)
3. Shared expanding path with cross-stream skip connections
4. Dense Block for message fusion              (Section 3.1.2)
5. ConvGRU iterative optimisation              (Section 3.4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class MedianPool2d(nn.Module):
    """Global spatial median pooling → (N, C, 1, 1)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        return x.view(N, C, -1).median(dim=-1).values.view(N, C, 1, 1)


def channel_shuffle(x: torch.Tensor, groups: int = 4) -> torch.Tensor:
    """ShuffleNet-style channel shuffle across `groups`."""
    N, C, H, W = x.shape
    assert C % groups == 0, f"C={C} must be divisible by groups={groups}"
    x = x.view(N, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(N, C, H, W)


class SharedMLP(nn.Module):
    """Two-layer 1×1 conv MLP used in channel attention.  C → C//r → C."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Multi-Scale Median Attention (MSMA)   –  Section 3.2
# ──────────────────────────────────────────────────────────────────────────────

class MSMAModule(nn.Module):
    """
    Multi-Scale Median Attention (MSMA) – Ji et al., 2025, Section 3.2.

    Three sub-modules applied in sequence:
      (a) Channel attention  : AvgPool + MaxPool + MedianPool → shared MLP → sum
      (b) Channel shuffle    : inter-channel information mixing (4 groups)
      (c) Spatial attention  : hierarchical depthwise convs (5×5, 1×7, 1×11) → 1×1 → multiply

    Input / Output: (N, C, H, W)  –  C must be divisible by 4.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # ── (a) channel attention ──────────────────────────────────────────
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.max_pool   = nn.AdaptiveMaxPool2d(1)
        self.med_pool   = MedianPool2d()
        self.mlp        = SharedMLP(channels, reduction)

        # ── (c) spatial attention ──────────────────────────────────────────
        self.dw_5x5  = nn.Conv2d(channels, channels, kernel_size=5,
                                 padding=2, groups=channels, bias=False)
        self.dw_1x7  = nn.Conv2d(channels, channels, kernel_size=(1, 7),
                                 padding=(0, 3), groups=channels, bias=False)
        self.dw_1x11 = nn.Conv2d(channels, channels, kernel_size=(1, 11),
                                 padding=(0, 5), groups=channels, bias=False)
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── (a) channel attention ──
        Fc = (self.mlp(self.avg_pool(x))
              + self.mlp(self.max_pool(x))
              + self.mlp(self.med_pool(x)))
        x_prime = Fc * x                         # F' = Fc ⊙ F

        # ── (b) channel shuffle ──
        x_shuf = channel_shuffle(x_prime, groups=4)   # F_shuffle

        # ── (c) spatial attention ──
        Fb = self.dw_5x5(x_shuf)
        Fs = self.dw_1x7(Fb) + self.dw_1x11(Fb)      # multi-scale sum
        spatial_map = self.spatial_conv(Fs)            # 1×1 conv
        x_out = torch.sigmoid(spatial_map) * x_prime  # F'' = σ(Fs) ⊙ F'

        return x_out


# ──────────────────────────────────────────────────────────────────────────────
# 2.  InceptionDMK Module  –  Section 3.3
# ──────────────────────────────────────────────────────────────────────────────

class InceptionDMKModule(nn.Module):
    """
    InceptionDMK – Inception-style Depthwise Multi-Kernel convolution.

    Four parallel branches (each receives full input, outputs ch//4 channels):
      • bypass   (x_pa)  : 1×1 pointwise
      • x_hw             : 3×3 depthwise-separable conv
      • x_w              : 1×11 depthwise-separable conv (horizontal)
      • x_h              : 11×1 depthwise-separable conv (vertical)

    Outputs are concatenated → ch channels (same as input).
    """

    def __init__(self, channels: int):
        super().__init__()
        branch_ch = channels // 4

        # bypass
        self.bypass = nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False)

        # 3×3 depthwise-separable
        self.dw_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
        )

        # 1×11 depthwise-separable (horizontal)
        self.dw_1x11 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5),
                      groups=channels, bias=False),
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
        )

        # 11×1 depthwise-separable (vertical)
        self.dw_11x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0),
                      groups=channels, bias=False),
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pa = self.bypass(x)
        x_hw = self.dw_3x3(x)
        x_w  = self.dw_1x11(x)
        x_h  = self.dw_11x1(x)
        out  = torch.cat([x_pa, x_hw, x_w, x_h], dim=1)  # → ch channels
        return self.bn(out)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  U-Net building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ContractingBlock(nn.Module):
    """One contracting step: Conv 3×3 → BN → LeakyReLU (+ optional MSMA)."""

    def __init__(self, in_ch: int, out_ch: int, use_msma: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.msma = MSMAModule(out_ch) if use_msma else None
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        """Returns (after_pool, skip_features)."""
        x = F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)
        if self.msma is not None:
            x = self.msma(x)
        skip = x                   # saved before pooling for skip connection
        x    = self.pool(x)
        return x, skip


class BottleneckBlock(nn.Module):
    """Bottleneck: Conv 3×3 → BN → LeakyReLU (+ optional MSMA, no pooling)."""

    def __init__(self, in_ch: int, out_ch: int, use_msma: bool = False,
                 use_inception: bool = False):
        super().__init__()
        self.conv      = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn        = nn.BatchNorm2d(out_ch)
        self.msma      = MSMAModule(out_ch)       if use_msma      else None
        self.inception = InceptionDMKModule(out_ch) if use_inception else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)
        if self.msma is not None:
            x = self.msma(x)
        if self.inception is not None:
            x = self.inception(x)
        return x


class ExpandingBlock(nn.Module):
    """One expanding step: TransposedConv 5×5 → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5,
                                         stride=2, padding=2, output_padding=1,
                                         bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.deconv(x)), inplace=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Dense Block for message fusion  –  Section 3.1.2
# ──────────────────────────────────────────────────────────────────────────────

class DenseMessageFusion(nn.Module):
    """
    Three densely-connected conv layers that fuse UNet output with message M.
    At every layer, both the accumulated features AND the original message M
    are re-injected to prevent information decay.

    in_ch = unet_out_channels + data_depth
    Each intermediate layer: 32 ch.  Final output: out_ch channels.
    """

    def __init__(self, unet_out_ch: int, data_depth: int, out_ch: int = 32):
        super().__init__()
        D = data_depth
        # Layer 1: [unet + M]
        self.conv1 = nn.Conv2d(unet_out_ch + D, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        # Layer 2: [x1 + unet + M]
        self.conv2 = nn.Conv2d(out_ch + unet_out_ch + D, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        # Layer 3: [x1 + x2 + unet + M]
        self.conv3 = nn.Conv2d(out_ch * 2 + unet_out_ch + D, out_ch, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(out_ch)

    def forward(self, unet_feat: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """unet_feat: (N, unet_out_ch, H, W), msg: (N, D, H, W) → (N, 32, H, W)"""
        x_in = torch.cat([unet_feat, msg], dim=1)

        x1 = F.leaky_relu(self.bn1(self.conv1(x_in)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(torch.cat([x1, x_in], dim=1))), 0.2, inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(torch.cat([x1, x2, x_in], dim=1))), 0.2, inplace=True)
        return x3


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Convolutional GRU for iterative optimisation  –  Section 3.4
# ──────────────────────────────────────────────────────────────────────────────

class ConvGRUCell(nn.Module):
    """
    2-D Convolutional GRU cell.  All operations are spatial (3×3 convolutions).

    xt: input  (N, input_ch, H, W)
    ht: hidden (N, hidden_ch, H, W)
    """

    def __init__(self, input_ch: int, hidden_ch: int):
        super().__init__()
        total = input_ch + hidden_ch
        self.conv_z = nn.Conv2d(total, hidden_ch, kernel_size=3, padding=1)
        self.conv_r = nn.Conv2d(total, hidden_ch, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(total, hidden_ch, kernel_size=3, padding=1)

    def forward(self, xt: torch.Tensor,
                ht: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([ht, xt], dim=1)
        z = torch.sigmoid(self.conv_z(combined))          # update gate
        r = torch.sigmoid(self.conv_r(combined))          # reset gate
        h_tilde = torch.tanh(self.conv_h(
            torch.cat([r * ht, xt], dim=1)))              # candidate state
        ht_new = (1 - z) * ht + z * h_tilde
        return ht_new


class PerturbationNetwork(nn.Module):
    """Converts GRU hidden state → 3-channel perturbation update direction."""

    def __init__(self, hidden_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_ch // 2, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Edge-Guided Dual-Stream U-Net Encoder  (main class)
# ──────────────────────────────────────────────────────────────────────────────

class EdgeGuidedDualStreamUNetEncoder(nn.Module):
    """
    Edge-Guided Dual-Stream U-Net for Secure Image Steganography.
    Ji, Zhang, Lv – Applied Sciences, 2025.

    Architecture (Table 1 + Sections 3.1 – 3.4):
    ────────────────────────────────────────────────────────────────────────
    Dual-stream contracting path
      Original stream (C): G1C–G5C, each with MSMA attention
      Edge stream     (E): G1E–G4E plain conv, G5E with InceptionDMK

    Shared expanding path
      G6–G10, fusing edge & cover skip features at each resolution

    Dense Block (message fusion)
      Fuses UNet output with secret message M

    ConvGRU iterative optimisation  (T iterations)
      Refines perturbation δ; during training returns all T stego images
      for the weighted iterative loss

    Parameters
    ──────────────────────────────────────────────────────────────────────
    data_depth : bits per pixel (D)
    T          : number of iterative GRU optimisation steps (default 10)
    eta        : step-size learning rate for perturbation update
    gamma      : loss decay factor (0 < γ ≤ 1, default 0.8)
    alpha      : weight for image-quality loss (MSE)
    sobel_alpha: edge enhancement strength α in Eq. 1 (default 1.0)
    hidden_ch  : hidden channels of the ConvGRU
    """

    def __init__(self, data_depth: int, T: int = 10, eta: float = 1.0,
                 gamma: float = 0.8, alpha: float = 1.0,
                 sobel_alpha: float = 1.0, hidden_ch: int = 32):
        super().__init__()
        self.version      = '1'
        self.data_depth   = data_depth
        self.T            = T
        self.eta          = eta
        self.gamma        = gamma
        self.alpha        = alpha
        self.sobel_alpha  = sobel_alpha

        # ── Sobel kernels (fixed, not learned) ──────────────────────────
        Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        Ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('Kx', Kx)
        self.register_buffer('Ky', Ky)

        # ── Dual-stream contracting path ─────────────────────────────────
        # cover stream:  G1C … G4C  (with MSMA), G5C (bottleneck + MSMA)
        self.G1C = ContractingBlock(3,   16,  use_msma=True)
        self.G2C = ContractingBlock(16,  32,  use_msma=True)
        self.G3C = ContractingBlock(32,  64,  use_msma=True)
        self.G4C = ContractingBlock(64,  128, use_msma=True)
        self.G5C = BottleneckBlock(128,  128, use_msma=True,  use_inception=False)

        # edge stream:  G1E … G4E (plain), G5E (bottleneck + InceptionDMK)
        self.G1E = ContractingBlock(3,   16,  use_msma=False)
        self.G2E = ContractingBlock(16,  32,  use_msma=False)
        self.G3E = ContractingBlock(32,  64,  use_msma=False)
        self.G4E = ContractingBlock(64,  128, use_msma=False)
        self.G5E = BottleneckBlock(128,  128, use_msma=False, use_inception=True)

        # ── Shared expanding path ─────────────────────────────────────────
        # G6: bottleneck → level-4  (H/16 → H/8)
        self.G6 = ExpandingBlock(128, 128)      # takes G5E output
        # After skip-1 fuse(G4E[128]+G6[128])+concat(G4C[128]) → 256
        self.G7 = ExpandingBlock(256, 64)       # H/8 → H/4
        # After skip-2 fuse(G3E[64]+G7[64])+concat(G3C[64])   → 128
        self.G8 = ExpandingBlock(128, 32)       # H/4 → H/2
        # After skip-3 fuse(G2E[32]+G8[32])+concat(G2C[32])   → 64
        self.G9 = ExpandingBlock(64, 16)        # H/2 → H
        # After skip-4 fuse(G1E[16]+G9[16])+concat(G1C[16])   → 32
        self.G10 = nn.Sequential(               # same resolution, ch 32 → 3
            nn.Conv2d(32, 3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # ── Dense Block for message fusion ───────────────────────────────
        # UNet outputs 3 ch; dense block fuses with D-ch message
        UNET_OUT = 3
        DENSE_OUT = 32
        self.dense_block = DenseMessageFusion(UNET_OUT, data_depth, out_ch=DENSE_OUT)

        # ── ConvGRU iterative optimisation ────────────────────────────────
        # GRU input: concat(δ[3], ∇δL[3], F[DENSE_OUT]) = 3+3+DENSE_OUT
        gru_in = 3 + 3 + DENSE_OUT
        self.gru_cell   = ConvGRUCell(gru_in, hidden_ch)
        self.perturb_net = PerturbationNetwork(hidden_ch)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def upgrade_legacy(self):
        if not hasattr(self, 'version'):
            self.version = '1'

    def _sobel_enhance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Create the edge-enhanced image E = C + α·‖∇C‖  (Eq. 1).
        Operates channel-wise (same kernel applied to each of R, G, B).
        image: (N, 3, H, W) in range [-1, 1]
        """
        N, C, H, W = image.shape
        # reshape to (N*3, 1, H, W) for a single shared filter
        x_gray = image.view(N * C, 1, H, W)
        Gx = F.conv2d(x_gray, self.Kx, padding=1)
        Gy = F.conv2d(x_gray, self.Ky, padding=1)
        grad_mag = torch.sqrt(Gx ** 2 + Gy ** 2 + 1e-8).view(N, C, H, W)
        enhanced = (image + self.sobel_alpha * grad_mag).clamp(-1.0, 1.0)
        return enhanced

    def _run_unet(self, cover: torch.Tensor,
                  edge: torch.Tensor):
        """
        Pass cover + edge-enhanced images through the dual-stream U-Net.

        Returns
        -------
        unet_out : (N, 3, H, W)   – G10 output
        """
        # Contracting path
        xC1, sC1 = self.G1C(cover)
        xC2, sC2 = self.G2C(xC1)
        xC3, sC3 = self.G3C(xC2)
        xC4, sC4 = self.G4C(xC3)
        bC       = self.G5C(xC4)     # bottleneck cover stream

        xE1, sE1 = self.G1E(edge)
        xE2, sE2 = self.G2E(xE1)
        xE3, sE3 = self.G3E(xE2)
        xE4, sE4 = self.G4E(xE3)
        bE       = self.G5E(xE4)     # bottleneck edge stream (+ InceptionDMK)

        # Expanding path  ─────────────────────────────────────────────────
        # G6: upsample from bottleneck
        g6 = self.G6(bE)                              # (N,128,H/8,W/8)

        # Skip 1: fuse G4E skip + G6, then concat G4C skip
        fus1    = sE4 + g6                            # element-wise add (both 128ch)
        concat1 = torch.cat([sC4, fus1], dim=1)       # → 256 ch

        # G7
        g7   = self.G7(concat1)                       # (N,64,H/4,W/4)
        fus2 = sE3 + g7
        concat2 = torch.cat([sC3, fus2], dim=1)       # → 128 ch

        # G8
        g8   = self.G8(concat2)                       # (N,32,H/2,W/2)
        fus3 = sE2 + g8
        concat3 = torch.cat([sC2, fus3], dim=1)       # → 64 ch

        # G9
        g9   = self.G9(concat3)                       # (N,16,H,W)
        fus4 = sE1 + g9
        concat4 = torch.cat([sC1, fus4], dim=1)       # → 32 ch

        # G10
        unet_out = self.G10(concat4)                  # (N,3,H,W)
        return unet_out

    def _iterative_optimise(self, cover: torch.Tensor,
                            features_F: torch.Tensor,
                            training: bool = False):
        """
        GRU-based iterative perturbation optimisation  (Section 3.4).

        Parameters
        ----------
        cover      : (N, 3, H, W)
        features_F : (N, DENSE_OUT, H, W)  – Dense Block output
        training   : if True, returns all T stego images; else only final

        Returns
        -------
        stego_list : list of T tensors (N,3,H,W)  if training=True
        stego      : (N,3,H,W)                    if training=False
        """
        N, _, H, W = cover.shape
        hidden_ch  = self.gru_cell.conv_z.out_channels

        delta = torch.zeros_like(cover)
        h_t   = torch.zeros(N, hidden_ch, H, W,
                            device=cover.device, dtype=cover.dtype)
        stego_list = []

        for t in range(self.T):
            # Compute loss gradient w.r.t. δ (used as a feature signal)
            if training and delta.requires_grad:
                # During training backward is available
                grad_delta = torch.autograd.grad(
                    (cover + delta).sum(), delta,
                    create_graph=True, retain_graph=True,
                    allow_unused=True
                )[0]
                if grad_delta is None:
                    grad_delta = torch.zeros_like(delta)
            else:
                grad_delta = torch.zeros_like(delta)

            # GRU input: concat(δ, ∇δL, F)
            x_t = torch.cat([delta, grad_delta, features_F], dim=1)

            # GRU update
            h_t    = self.gru_cell(x_t, h_t)

            # Perturbation update direction
            g_t    = self.perturb_net(h_t)
            delta  = delta + self.eta * g_t

            # Stego image at step t
            S_t    = (cover + delta).clamp(-1.0, 1.0)

            if training:
                stego_list.append(S_t)

        if training:
            return stego_list
        else:
            return S_t  # only final stego image for inference

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, image: torch.Tensor,
                data: torch.Tensor,
                training: bool = None) -> object:
        """
        Parameters
        ----------
        image    : cover image  (N, 3, H, W),  values in [-1, 1]
        data     : secret message (N, D, H, W), values in {0, 1}
        training : override for training/inference mode.
                   If None, uses self.training (set by model.train() / model.eval()).

        Returns
        -------
        training=True  → list of T stego tensors (N, 3, H, W)
        training=False → single stego tensor     (N, 3, H, W)
        """
        if training is None:
            training = self.training

        # ① Edge enhancement
        edge = self._sobel_enhance(image)

        # ② Dual-stream U-Net
        unet_out = self._run_unet(image, edge)        # (N, 3, H, W)

        # ③ Dense Block: fuse UNet features with secret message
        features_F = self.dense_block(unet_out, data)  # (N, 32, H, W)

        # ④ GRU iterative optimisation
        # Make delta differentiable during training
        if training:
            result = self._iterative_optimise(image, features_F, training=True)
        else:
            with torch.no_grad():
                result = self._iterative_optimise(image, features_F, training=False)

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Legacy encoders (SteganoGAN baselines)
# ──────────────────────────────────────────────────────────────────────────────

class BasicEncoder(nn.Module):
    """
    Basic encoder: cover → 32-ch features, concat message, two conv layers → stego.
    Input: (N, 3, H, W), (N, D, H, W)  Output: (N, 3, H, W)
    """

    def __init__(self, data_depth):
        super(BasicEncoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        self.feature_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.feature_bn   = nn.BatchNorm2d(32)
        self.layer1_conv  = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.layer1_bn    = nn.BatchNorm2d(32)
        self.layer2_conv  = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_bn    = nn.BatchNorm2d(32)
        self.output_conv  = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, image, data):
        x = F.leaky_relu(self.feature_bn(self.feature_conv(image)), inplace=True)
        x = F.leaky_relu(self.layer1_bn(self.layer1_conv(torch.cat([x, data], 1))), inplace=True)
        x = F.leaky_relu(self.layer2_bn(self.layer2_conv(x)), inplace=True)
        return torch.tanh(self.output_conv(x))


class ResidualEncoder(nn.Module):
    """Residual encoder: BasicEncoder + residual connection (C + output)."""

    def __init__(self, data_depth):
        super(ResidualEncoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        self.feature_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.feature_bn   = nn.BatchNorm2d(32)
        self.layer1_conv  = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.layer1_bn    = nn.BatchNorm2d(32)
        self.layer2_conv  = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_bn    = nn.BatchNorm2d(32)
        self.output_conv  = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, image, data):
        x = F.leaky_relu(self.feature_bn(self.feature_conv(image)), inplace=True)
        x = F.leaky_relu(self.layer1_bn(self.layer1_conv(torch.cat([x, data], 1))), inplace=True)
        x = F.leaky_relu(self.layer2_bn(self.layer2_conv(x)), inplace=True)
        return image + self.output_conv(x)


class DenseEncoder(nn.Module):
    """DenseNet-style encoder with dense skip connections (SteganoGAN default)."""

    def __init__(self, data_depth):
        super(DenseEncoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(96 + self.data_depth, 3, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, image, data):
        x1 = F.leaky_relu(self.bn1(self.conv1(image)), inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(torch.cat([x1, data], 1))), inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(torch.cat([x1, x2, data], 1))), inplace=True)
        x4 = self.conv4(torch.cat([x1, x2, x3, data], 1))
        return image + x4
