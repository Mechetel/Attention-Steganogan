

import gc
import torch
from torch.optim import Adam
from tqdm import tqdm

from ..utils import ssim
from ..metrics_calculator import MetricsCalculator
from ..generators.payload_generator import PayloadGenerator


class TrainingManager:
    """Manages the training process."""

    def __init__(self, encoder, decoder, data_depth, device, verbose=False):
        self.encoder = encoder
        self.decoder = decoder
        self.data_depth = data_depth
        self.device = device
        self.verbose = verbose
        self.optimizer = None
        self.payload_generator = PayloadGenerator(device)

    def get_optimizer(self):
        """Create and return optimizer for encoder and decoder."""
        _enc_dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        return Adam(_enc_dec_list, lr=1e-4)

    def encode_decode(self, cover, quantize=False):
        """Perform encoding and decoding."""
        payload = self.payload_generator.random_data(cover, self.data_depth)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)
        return generated, payload, decoded

    def fit_coders(self, train, metrics):
        """Train encoder and decoder for one epoch."""
        pbar = tqdm(train, disable=not self.verbose, desc="Training",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for cover, _ in pbar:
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self.encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = MetricsCalculator.coding_scores(
                cover, generated, payload, decoded
            )

            self.optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss).backward()
            self.optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

            pbar.set_postfix({
                'enc_mse': f'{encoder_mse.item():.4f}',
                'dec_loss': f'{decoder_loss.item():.4f}',
                'dec_acc': f'{decoder_acc.item():.4f}'
            })

    def validate(self, validate, metrics):
        """Validate the model."""
        pbar = tqdm(validate, disable=not self.verbose, desc="Validation",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for cover, _ in pbar:
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self.encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = MetricsCalculator.coding_scores(
                cover, generated, payload, decoded
            )

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.ssim'].append(ssim(cover, generated).item())
            metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
            metrics['val.rsbpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

            pbar.set_postfix({
                'ssim': f'{metrics["val.ssim"][-1]:.4f}',
                'psnr': f'{metrics["val.psnr"][-1]:.2f}',
                'rsbpp': f'{metrics["val.rsbpp"][-1]:.4f}'
            })