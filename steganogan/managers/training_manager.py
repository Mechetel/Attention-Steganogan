

import gc
import torch
from torch.optim import Adam
from tqdm import tqdm

from ..utils import ssim
from ..metrics_calculator import MetricsCalculator
from ..generators.payload_generator import PayloadGenerator


class TrainingManager:
    """Manages the training process with optional critic (adversarial) training."""

    def __init__(self, encoder, decoder, data_depth, device, verbose=False,
                 critic=None, critic_train_steps=5):
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.data_depth = data_depth
        self.device = device
        self.verbose = verbose
        self.optimizer = None
        self.critic_optimizer = None
        self.critic_train_steps = critic_train_steps
        self.payload_generator = PayloadGenerator(device)

    @property
    def use_critic(self):
        """Whether adversarial training with critic is enabled."""
        return self.critic is not None

    def _critic(self, image):
        """Evaluate the image using the critic."""
        return torch.mean(self.critic(image))

    def get_optimizer(self):
        """Create and return optimizer for encoder and decoder."""
        _enc_dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        return Adam(_enc_dec_list, lr=1e-4)

    def get_critic_optimizer(self):
        """Create and return optimizer for critic."""
        if self.critic is None:
            return None
        return Adam(self.critic.parameters(), lr=1e-4)

    def encode_decode(self, cover, quantize=False):
        """Perform encoding and decoding."""
        payload = self.payload_generator.random_data(cover, self.data_depth)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)
        return generated, payload, decoded

    def _fit_critic_step(self, cover):
        """Train critic for one step on a batch."""
        gc.collect()
        payload = self.payload_generator.random_data(cover, self.data_depth)
        generated = self.encoder(cover, payload)

        cover_score = self._critic(cover)
        generated_score = self._critic(generated)

        self.critic_optimizer.zero_grad()
        # WGAN: critic wants to maximize (cover_score - generated_score)
        # So we minimize (generated_score - cover_score) = fake - real
        (generated_score - cover_score).backward(retain_graph=False)
        self.critic_optimizer.step()

        # WGAN weight clipping for Lipschitz constraint
        for p in self.critic.parameters():
            p.data.clamp_(-0.1, 0.1)

        return cover_score.item(), generated_score.item()

    def fit_coders(self, train, metrics):
        """Train encoder, decoder, and optionally critic for one epoch."""
        pbar = tqdm(train, disable=not self.verbose, desc="Training",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for cover, _ in pbar:
            gc.collect()
            cover = cover.to(self.device)

            # === Critic training (if enabled) ===
            if self.use_critic:
                for _ in range(self.critic_train_steps):
                    cover_score, generated_score = self._fit_critic_step(cover)

                metrics['train.cover_score'].append(cover_score)
                metrics['train.generated_score'].append(generated_score)

            # === Encoder + Decoder training ===
            generated, payload, decoded = self.encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = MetricsCalculator.coding_scores(
                cover, generated, payload, decoded
            )

            if self.use_critic:
                generated_score = self._critic(generated)

                self.optimizer.zero_grad()
                # Encoder wants critic to score generated images high (fool critic)
                # Critic is trained to score generated LOW, so encoder minimizes -generated_score
                (100.0 * encoder_mse + decoder_loss - generated_score).backward()
                self.optimizer.step()
            else:
                self.optimizer.zero_grad()
                (100.0 * encoder_mse + decoder_loss).backward()
                self.optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

            postfix = {
                'enc_mse': f'{encoder_mse.item():.4f}',
                'dec_loss': f'{decoder_loss.item():.4f}',
                'dec_acc': f'{decoder_acc.item():.4f}'
            }
            if self.use_critic:
                postfix['cover'] = f'{metrics["train.cover_score"][-1]:.4f}'
                postfix['gen'] = f'{metrics["train.generated_score"][-1]:.4f}'
            pbar.set_postfix(postfix)

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

            if self.use_critic:
                with torch.no_grad():
                    cover_score = self._critic(cover)
                    generated_score = self._critic(generated)
                metrics['val.cover_score'].append(cover_score.item())
                metrics['val.generated_score'].append(generated_score.item())

            pbar.set_postfix({
                'ssim': f'{metrics["val.ssim"][-1]:.4f}',
                'psnr': f'{metrics["val.psnr"][-1]:.2f}',
                'rsbpp': f'{metrics["val.rsbpp"][-1]:.4f}'
            })