import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from typing import Dict, Tuple
from .utils import ssim


class MetricsCalculator:
    """Calculates and manages training metrics."""

    @staticmethod
    def coding_scores(
        cover: torch.Tensor,
        generated: torch.Tensor,
        payload: torch.Tensor,
        decoded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute encoder MSE, decoder BCE loss, and decoder bit accuracy.

        Returns
        -------
        encoder_mse  : pixel-level MSE between stego and cover image
        decoder_loss : binary cross-entropy between decoded and payload bits
        decoder_acc  : fraction of bits decoded correctly
        """
        encoder_mse  = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc  = (
            (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
        )
        return encoder_mse, decoder_loss, decoder_acc

    @staticmethod
    def calculate_validation_metrics(
        cover: torch.Tensor,
        generated: torch.Tensor,
        payload: torch.Tensor,
        decoded: torch.Tensor,
        data_depth: int,
    ) -> Dict[str, float]:
        """
        Compute the full suite of validation metrics.

        Returns a dict with keys: encoder_mse, decoder_loss, decoder_acc,
        ssim, psnr, rsbpp.
        """
        encoder_mse, decoder_loss, decoder_acc = MetricsCalculator.coding_scores(
            cover, generated, payload, decoded
        )
        return {
            'encoder_mse' : encoder_mse.item(),
            'decoder_loss': decoder_loss.item(),
            'decoder_acc' : decoder_acc.item(),
            'ssim'        : ssim(cover, generated).item(),
            'psnr'        : 10 * torch.log10(4 / encoder_mse).item(),
            'rsbpp'       : data_depth * (2 * decoder_acc.item() - 1),
        }
