# -*- coding: utf-8 -*-
"""
Trainer: manages one full epoch of encoder/decoder (+ optional critic) training.

Supports two encoder modes transparently:
  Standard   — encoder.forward() returns a single (N,3,H,W) tensor.
  Iterative  — encoder.forward() returns a list of T stego tensors.
               Detected automatically via isinstance check.

Mixed-precision (AMP) is enabled automatically on CUDA devices.
Combined with gradient checkpointing in DenseASPP / EdgeAwareDenseDecoder /
EdgeAwareDenseMessageFusion, this halves activation memory and allows
batch_size=8, T=8 at crop_size=360 on an 11 GB GPU.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .losses  import SteganographyLoss, IterativeLoss, EdgeAwareIterativeLoss
from .metrics import SteganographyMetrics
from ..utils.payload      import PayloadFactory
from ..utils.image_quality import SSIMCalculator
from ..models.critics.critics import MultiScaleEdgeAwareCritic


def _is_iterative(output) -> bool:
    """True if encoder output is a list/tuple of stego tensors."""
    return isinstance(output, (list, tuple))



class Trainer:
    """
    Encapsulates one training epoch plus validation.

    Parameters
    ----------
    encoder           : encoder module
    decoder           : decoder module
    data_depth        : bits per pixel
    device            : compute device
    verbose           : show tqdm progress bars
    critic            : optional WGAN critic module
    critic_train_steps: number of critic update steps per encoder step
    """

    def __init__(
        self,
        encoder:            nn.Module,
        decoder:            nn.Module,
        data_depth:         int,
        device:             torch.device,
        verbose:            bool = False,
        critic:             Optional[nn.Module] = None,
        critic_train_steps: int = 5,
    ) -> None:
        self.encoder            = encoder
        self.decoder            = decoder
        self.critic             = critic
        self.data_depth         = data_depth
        self.device             = device
        self.verbose            = verbose
        # Spectral-norm critics enforce Lipschitz per-layer — no repeated
        # critic steps needed (unlike weight-clipping WGAN which needs 5).
        self.critic_train_steps = (
            1 if isinstance(critic, MultiScaleEdgeAwareCritic)
            else critic_train_steps
        )

        self.optimizer:        Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None

        self._payload_factory = PayloadFactory(device)
        self._std_loss        = SteganographyLoss(encoder_weight=100.0)
        self._iter_loss       = IterativeLoss(
            decoder=decoder,
            gamma=getattr(encoder, "gamma", 0.8),
            alpha=getattr(encoder, "alpha", 100.0),
            critic=critic,
        )

        # Edge-aware iterative loss (for EdgeAwareDenseASPPEncoder)
        self._edge_iter_loss = EdgeAwareIterativeLoss(
            decoder=decoder,
            gamma=getattr(encoder, "gamma", 0.8),
            alpha=getattr(encoder, "alpha", 100.0),
            critic=critic,
            lambda_edge=getattr(encoder, "lambda_edge", 0.01),
            lambda_vgg=getattr(encoder, "lambda_vgg", 0.1),
        ).to(device)

        # Detect if critic uses spectral norm (no weight clipping needed)
        self._critic_uses_spectral_norm = isinstance(critic, MultiScaleEdgeAwareCritic)

        # AMP: enabled on CUDA, no-op on CPU/MPS.
        self._amp_device = device.type if isinstance(device, torch.device) else device
        self._use_amp    = (self._amp_device == "cuda")
        self._scaler     = GradScaler(device=self._amp_device) if self._use_amp else None
        self._critic_scaler = GradScaler(device=self._amp_device) if self._use_amp else None

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def has_critic(self) -> bool:
        return self.critic is not None

    # ── Optimiser factories ───────────────────────────────────────────────────

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Adam over encoder + decoder parameters."""
        return Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-4,
        )

    def build_critic_optimizer(self) -> Optional[torch.optim.Optimizer]:
        """Adam over critic parameters, or None if no critic."""
        return Adam(self.critic.parameters(), lr=1e-4) if self.has_critic else None

    # Backward-compat aliases used by engine.py
    def get_optimizer(self): return self.build_optimizer()
    def get_critic_optimizer(self): return self.build_critic_optimizer()

    # ── Critic training step ──────────────────────────────────────────────────

    def _critic_score(self, image: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.critic(image))

    def _train_critic_step(
        self, cover: torch.Tensor
    ) -> tuple:
        """One WGAN critic gradient step.  Returns (cover_score, gen_score)."""
        payload = self._payload_factory.random(cover, self.data_depth)
        with torch.no_grad():
            raw_out = self.encoder(cover, payload)
        stego = (raw_out[-1] if _is_iterative(raw_out) else raw_out).detach()

        with autocast(device_type=self._amp_device, enabled=self._use_amp):
            cover_score = self._critic_score(cover)
            gen_score   = self._critic_score(stego)
            critic_loss = gen_score - cover_score

        self.critic_optimizer.zero_grad()
        if self._use_amp:
            self._critic_scaler.scale(critic_loss).backward()
            self._critic_scaler.step(self.critic_optimizer)
            self._critic_scaler.update()
        else:
            critic_loss.backward()
            self.critic_optimizer.step()

        if not self._critic_uses_spectral_norm:
            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

        return cover_score.item(), gen_score.item()

    # ── Encode + decode ───────────────────────────────────────────────────────

    def _encode_decode(self, cover: torch.Tensor, quantize: bool = False):
        """
        Forward pass through encoder and decoder.

        Returns (stego, payload, decoded_logits, raw_encoder_output).
        """
        payload = self._payload_factory.random(cover, self.data_depth)
        raw_out = self.encoder(cover, payload)
        stego   = raw_out[-1] if _is_iterative(raw_out) else raw_out

        if quantize:
            stego = 2.0 * ((255.0 * (stego + 1.0) / 2.0).long().float()) / 255.0 - 1.0

        decoded = self.decoder(stego)
        return stego, payload, decoded, raw_out

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(
        self,
        loader:  torch.utils.data.DataLoader,
        metrics: Dict[str, List[float]],
    ) -> None:
        """Run one training epoch, appending per-batch values to *metrics*."""
        pbar = tqdm(
            loader, disable=not self.verbose,
            desc="Train",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for cover, _ in pbar:
            cover = cover.to(self.device)

            # ── Critic update ────────────────────────────────────────────
            if self.has_critic:
                for _ in range(self.critic_train_steps):
                    cv_s, gn_s = self._train_critic_step(cover)
                metrics["train.cover_score"].append(cv_s)
                metrics["train.generated_score"].append(gn_s)

            # ── Encoder + decoder update ─────────────────────────────────
            with autocast(device_type=self._amp_device, enabled=self._use_amp):
                stego, payload, decoded, raw_out = self._encode_decode(cover)

                if _is_iterative(raw_out):
                    edge_map = getattr(self.encoder, "_last_edge_map", None)
                    if edge_map is not None:
                        loss = self._edge_iter_loss(cover, payload, raw_out, edge_map)
                    else:
                        loss = self._iter_loss(cover, payload, raw_out)
                else:
                    gen_score = (self._critic_score(stego) if self.has_critic else None)
                    loss = self._std_loss(cover, stego, payload, decoded, gen_score)

            with torch.no_grad():
                enc_mse, dec_loss, dec_acc = SteganographyMetrics.coding_scores(
                    cover, stego, payload, decoded
                )

            self.optimizer.zero_grad()
            if self._use_amp:
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            metrics["train.encoder_mse"].append(enc_mse.item())
            metrics["train.decoder_loss"].append(dec_loss.item())
            metrics["train.decoder_acc"].append(dec_acc.item())

            pbar.set_postfix(
                enc_mse=f"{enc_mse.item():.4f}",
                dec_loss=f"{dec_loss.item():.4f}",
                dec_acc=f"{dec_acc.item():.4f}",
            )

    # ── Validation epoch ──────────────────────────────────────────────────────

    def validate_epoch(
        self,
        loader:  torch.utils.data.DataLoader,
        metrics: Dict[str, List[float]],
    ) -> None:
        """Run one validation pass (no gradients), appending values to *metrics*."""
        pbar = tqdm(
            loader, disable=not self.verbose,
            desc="Val  ",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        with torch.no_grad():
            for cover, _ in pbar:
                cover = cover.to(self.device)
                stego, payload, decoded, _ = self._encode_decode(cover, quantize=True)

                enc_mse, dec_loss, dec_acc = SteganographyMetrics.coding_scores(
                    cover, stego, payload, decoded
                )

                metrics["val.encoder_mse"].append(enc_mse.item())
                metrics["val.decoder_loss"].append(dec_loss.item())
                metrics["val.decoder_acc"].append(dec_acc.item())
                metrics["val.ssim"].append(SSIMCalculator.calculate(cover, stego).item())
                metrics["val.psnr"].append(10.0 * torch.log10(4.0 / enc_mse).item())
                metrics["val.rsbpp"].append(self.data_depth * (2.0 * dec_acc.item() - 1.0))

                if self.has_critic:
                    metrics["val.cover_score"].append(self._critic_score(cover).item())
                    metrics["val.generated_score"].append(self._critic_score(stego).item())

                pbar.set_postfix(
                    ssim=f"{metrics['val.ssim'][-1]:.4f}",
                    psnr=f"{metrics['val.psnr'][-1]:.2f}",
                    rsbpp=f"{metrics['val.rsbpp'][-1]:.4f}",
                )

    # ── Backward-compat aliases ───────────────────────────────────────────────

    def fit_coders(self, loader, metrics): self.train_epoch(loader, metrics)
    def validate(self, loader, metrics):   self.validate_epoch(loader, metrics)
