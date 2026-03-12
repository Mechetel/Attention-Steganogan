# -*- coding: utf-8 -*-
"""
Steganography loss functions.

  SteganographyLoss : standard single-step loss (encoder MSE + decoder BCE ± critic)
  IterativeLoss     : weighted multi-step loss for iterative encoders (Ji et al., Eq. 18)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SteganographyLoss:
    """
    Standard single-step steganography loss.

    L_total = λ_enc · MSE(S, C) + BCE(M', M) [- critic_score(S)]

    Parameters
    ----------
    encoder_weight : MSE loss scale factor (default 100)
    """

    def __init__(self, encoder_weight: float = 100.0) -> None:
        self.encoder_weight = encoder_weight

    def __call__(
        self,
        cover:      torch.Tensor,
        stego:      torch.Tensor,
        payload:    torch.Tensor,
        decoded:    torch.Tensor,
        gen_score:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cover, stego    : (N,3,H,W) image tensors
        payload         : (N,D,H,W) ground-truth bits
        decoded         : (N,D,H,W) decoder output logits
        gen_score       : scalar critic score for *stego* (WGAN, optional)
        """
        enc_loss = F.mse_loss(stego, cover)
        dec_loss = F.binary_cross_entropy_with_logits(decoded, payload.float())
        total    = self.encoder_weight * enc_loss + dec_loss
        if gen_score is not None:
            total = total - gen_score
        return total


class IterativeLoss:
    """
    Weighted iterative loss for ConvGRU-based encoders (Eq. 18, Ji et al. 2025).

    L_total = Σ_{t=0}^{T-1}  γ^{T-1-t} · [L_D(M, M'_t) + α·L_E(C, S_t) + β·L_C(C, S_t)]

      L_D : BCE between recovered bits M'_t and ground-truth M
      L_E : pixel-level MSE  (image quality)
      L_C : Wasserstein proxy  critic(S_t) – critic(C)   [optional]

    More recent steps receive higher weight (γ^0 = 1 at t = T-1).

    Parameters
    ----------
    decoder    : decoder module (produces logits from a stego image)
    gamma      : per-step discount factor  (0 < γ ≤ 1, default 0.8)
    alpha      : image-quality loss weight (default 1.0)
    critic     : optional critic module for adversarial loss term
    """

    def __init__(
        self,
        decoder: nn.Module,
        gamma:   float = 0.8,
        alpha:   float = 1.0,
        critic:  Optional[nn.Module] = None,
    ) -> None:
        self.decoder = decoder
        self.gamma   = gamma
        self.alpha   = alpha
        self.critic  = critic

    def __call__(
        self,
        cover:      torch.Tensor,
        payload:    torch.Tensor,
        stego_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cover      : (N,3,H,W) cover image
        payload    : (N,D,H,W) ground-truth bits
        stego_list : list of T stego tensors from the iterative encoder
        """
        T     = len(stego_list)
        total = torch.tensor(0.0, device=cover.device)

        # critic(cover) is the same for every step; compute it once under
        # no_grad (cover is a fixed input, not a learnable parameter).
        critic_cover_score: Optional[torch.Tensor] = None
        if self.critic is not None:
            with torch.no_grad():
                critic_cover_score = torch.mean(self.critic(cover))

        for t, S_t in enumerate(stego_list):
            weight  = self.gamma ** (T - 1 - t)     # most recent → weight 1

            M_prime = self.decoder(S_t)
            L_D = F.binary_cross_entropy_with_logits(M_prime, payload.float())
            L_E = F.mse_loss(S_t, cover)

            L_C = torch.tensor(0.0, device=cover.device)
            if self.critic is not None:
                L_C = torch.mean(self.critic(S_t)) - critic_cover_score

            total = total + weight * (L_D + self.alpha * L_E + L_C)

        return total
