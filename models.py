# -*- coding: utf-8 -*-
import gc
import inspect
import json
import os
from collections import Counter

import imageio
from PIL import Image

import torch
from imageio import imread, imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm

from utils import bits_to_bytearray, bytearray_to_text, ssim, text_to_bits

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train'
)

METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.ssim',
    'val.psnr',
    'val.rsbpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
]


class DeviceManager:
    """Manages device selection and model placement."""

    def __init__(self, gpu=True, verbose=False):
        self.verbose = verbose
        self.gpu_requested = gpu
        self.gpu = False
        self.device = None
        self.set_device()

    def set_device(self):
        if self.gpu_requested:
            if torch.cuda.is_available():
                self.gpu = True
                self.device = torch.device('cuda')
                if self.verbose:
                    print("Using NVIDIA GPU (CUDA).")
            elif torch.backends.mps.is_available():
                self.gpu = True
                self.device = torch.device('mps')
                if self.verbose:
                    print("Using Apple GPU (MPS).")
            else:
                self.gpu = False
                self.device = torch.device('cpu')
                if self.verbose:
                    print("GPU requested but not available. Falling back to CPU.")
        else:
            self.gpu = False
            self.device = torch.device('cpu')

            if self.verbose:
                print("Using CPU.")

    def to_device(self, *models):
        """Move models to the current device."""
        for model in models:
            model.to(self.device)


class PayloadGenerator:
    """Handles payload generation for encoding."""

    def __init__(self, device):
        self.device = device

    def random_data(self, cover, data_depth):
        """Generate random binary payload matching cover dimensions."""
        N, _, H, W = cover.size()
        return torch.zeros((N, data_depth, H, W), device=self.device).random_(0, 2)

    def make_payload(self, width, height, depth, text):
        """Create payload from text message."""
        message = text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)


class MetricsCalculator:
    """Calculates and manages training metrics."""

    @staticmethod
    def coding_scores(cover, generated, payload, decoded):
        """Calculate encoder and decoder performance metrics."""
        encoder_mse = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
        return encoder_mse, decoder_loss, decoder_acc

    @staticmethod
    def calculate_validation_metrics(cover, generated, payload, decoded, data_depth):
        """Calculate comprehensive validation metrics."""
        encoder_mse, decoder_loss, decoder_acc = MetricsCalculator.coding_scores(
            cover, generated, payload, decoded
        )

        metrics = {
            'encoder_mse': encoder_mse.item(),
            'decoder_loss': decoder_loss.item(),
            'decoder_acc': decoder_acc.item(),
            'ssim': ssim(cover, generated).item(),
            'psnr': 10 * torch.log10(4 / encoder_mse).item(),
            'rsbpp': data_depth * (2 * decoder_acc.item() - 1)
        }
        return metrics


class SampleGenerator:
    """Generates sample images during training."""

    def __init__(self, encoder, payload_generator, device):
        self.encoder = encoder
        self.payload_generator = payload_generator
        self.device = device

    def generate_samples(self, samples_path, epoch, text_to_encode, data_depth):
        """Generate and save sample encoded images."""
        callback_images_path = os.path.join('data', 'callback_images')
        if not os.path.exists(callback_images_path):
            os.makedirs(callback_images_path)
            raise ValueError("callback_images directory not found. Please add images to generate samples.")
        image_filenames = sorted(os.listdir(callback_images_path))
        if len(image_filenames) < 8:
            raise ValueError("Expected at least 8 generated images in callback_images")

        reshaped_tensors = []
        original_images = []

        for filename in image_filenames[:8]:
            path = os.path.join(callback_images_path, filename)

            image = imread(path, pilmode='RGB') / 127.5 - 1.0
            tensor = torch.FloatTensor(image).permute(2, 0, 1)

            if text_to_encode:
                cover = tensor.unsqueeze(0).to(self.device)
                cover_size = cover.size()

                payload = self.payload_generator.make_payload(
                    cover_size[3], cover_size[2], data_depth, text_to_encode
                )
                payload = payload.to(self.device)

                encoded_tensor = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)
                tensor = encoded_tensor.squeeze(0).to(self.device)

            original_tensor = tensor.clamp(-1.0, 1.0)
            original_tensor = ((original_tensor + 1.0) / 2.0 * 255.0).byte()
            original_image = Image.fromarray(original_tensor.permute(1, 2, 0).cpu().numpy())
            original_images.append((filename, original_image))

            resized_tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(360, 360), mode='bilinear', align_corners=False
            ).squeeze(0)
            reshaped_tensors.append(resized_tensor)

        self._create_and_save_grid(reshaped_tensors, samples_path, epoch)

    def _create_and_save_grid(self, reshaped_tensors, samples_path, epoch):
        """Create and save image grid."""
        batch = torch.stack(reshaped_tensors).clamp(-1.0, 1.0)
        batch = ((batch + 1.0) / 2.0 * 255.0).byte()
        images = [Image.fromarray(t.permute(1, 2, 0).cpu().numpy()) for t in batch]

        grid_cols = 4
        grid_rows = 2
        gap = 20
        img_w, img_h = images[0].size
        total_w = grid_cols * img_w + (grid_cols - 1) * gap
        total_h = grid_rows * img_h + (grid_rows - 1) * gap
        grid_img = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))

        for idx, img in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * (img_w + gap)
            y = row * (img_h + gap)
            grid_img.paste(img, (x, y))

        grid_filename = f'grid_epoch_{epoch}.png'
        grid_output_path = os.path.join(samples_path, grid_filename)
        grid_img.save(grid_output_path)


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


class HistoryManager:
    """Manages training history and metrics logging."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.history = []
        self.metrics_path = os.path.join(log_dir, 'metrics.log') if log_dir else None
        self._load_existing_history()

    def _load_existing_history(self):
        """Load existing metrics from log file."""
        if self.metrics_path and os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as metrics_file:
                    self.history = json.load(metrics_file)
                    print(f"Loaded {len(self.history)} previous epochs")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load existing metrics.log: {e}")
                self.history = []

    def append_and_save(self, metrics):
        """Append metrics to history and save to file."""
        self.history.append(metrics)
        if self.metrics_path:
            with open(self.metrics_path, 'w') as metrics_file:
                json.dump(self.history, metrics_file, indent=4)

    def print_metrics(self, metrics):
        """Print formatted metrics."""
        print(f"\nEpoch {metrics['epoch']} Metrics:")
        print(f"  Train - Enc MSE: {metrics['train.encoder_mse']:.6f}, "
              f"Dec Loss: {metrics['train.decoder_loss']:.4f}, "
              f"Dec Acc: {metrics['train.decoder_acc']:.4f}")
        print(f"  Val   - Enc MSE: {metrics['val.encoder_mse']:.6f}, "
              f"Dec Loss: {metrics['val.decoder_loss']:.4f}, "
              f"Dec Acc: {metrics['val.decoder_acc']:.4f}")
        print(f"  Val   - SSIM: {metrics['val.ssim']:.4f}, "
              f"PSNR: {metrics['val.psnr']:.2f}, "
              f"RSBPP: {metrics['val.rsbpp']:.4f}")


class EncoderManager:
    """Manages image encoding operations."""

    def __init__(self, encoder, payload_generator, data_depth, device, verbose=False):
        self.encoder = encoder
        self.payload_generator = payload_generator
        self.data_depth = data_depth
        self.device = device
        self.verbose = verbose

    def encode(self, cover_path, output_path, text):
        """Encode text message into cover image."""
        cover = imread(cover_path, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover_size = cover.size()
        payload = self.payload_generator.make_payload(
            cover_size[3], cover_size[2], self.data_depth, text
        )

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output_path, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')


class DecoderManager:
    """Manages image decoding operations."""

    def __init__(self, decoder, device, verbose=False):
        self.decoder = decoder
        self.device = device
        self.verbose = verbose

    def decode(self, image_path):
        """Decode hidden message from image."""
        if not os.path.exists(image_path):
            raise ValueError('Unable to read %s.' % image_path)

        image = imread(image_path, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0

        candidates = Counter()
        bits = image.data.int().cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        if len(candidates) == 0:
            raise ValueError('Failed to find message.')

        candidate, count = candidates.most_common(1)[0]

        if self.verbose:
            print(f'Decoding completed. Message found {count} times.')

        return candidate


class ModelLoader:
    """Handles model loading and initialization."""

    @staticmethod
    def load(path, gpu=True, verbose=False, log_dir=None):
        """Load model from file."""
        if path is None:
            raise ValueError('Please provide a path to pretrained model.')

        # Determine device for loading using same logic as DeviceManager
        if gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
        
        if verbose:
            print("Loading model to device:", device)

        steganogan = torch.load(path, map_location=device, weights_only=False)
        steganogan.verbose = verbose

        steganogan.encoder_decoder_optimizer = None
        steganogan.fit_metrics = None
        steganogan.history = list()

        steganogan.log_dir = log_dir
        if log_dir:
            os.makedirs(steganogan.log_dir, exist_ok=True)
            steganogan.samples_path = os.path.join(steganogan.log_dir, 'samples')
            os.makedirs(steganogan.samples_path, exist_ok=True)

        steganogan.encoder.upgrade_legacy()
        steganogan.decoder.upgrade_legacy()

        # Reinitialize device manager with gpu parameter
        steganogan.device_manager = DeviceManager(gpu=gpu, verbose=verbose)
        steganogan.device = steganogan.device_manager.device
        steganogan.gpu = steganogan.device_manager.gpu
        steganogan.device_manager.to_device(steganogan.encoder, steganogan.decoder)

        if verbose:
            print(f'Model loaded from {path}')

        return steganogan


class SteganoGAN(object):
    """Main SteganoGAN model class."""

    def __init__(self, data_depth, encoder, decoder, gpu=True, verbose=False, log_dir=None, **kwargs):
        self.verbose = verbose
        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth

        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)

        self.device_manager = DeviceManager(gpu=gpu, verbose=verbose)
        self.device = self.device_manager.device
        self.gpu = self.device_manager.gpu
        self.device_manager.to_device(self.encoder, self.decoder)

        self.payload_generator = PayloadGenerator(self.device)
        self.training_manager = TrainingManager(
            self.encoder, self.decoder, self.data_depth, self.device, verbose
        )

        self.encoder_decoder_optimizer = None
        self.fit_metrics = None
        self.history = list()

        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)
            self.history_manager = HistoryManager(log_dir)
        else:
            self.history_manager = None

        self.encoder_manager = EncoderManager(
            self.encoder, self.payload_generator, self.data_depth, self.device, verbose
        )
        self.decoder_manager = DecoderManager(self.decoder, self.device, verbose)

    def _get_instance(self, class_or_instance, kwargs):
        """Create instance from class or return existing instance."""
        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)

    def set_device(self):
        """Set compute device (automatically detects CUDA > MPS > CPU)."""
        self.device_manager.set_device()
        self.device = self.device_manager.device
        self.gpu = self.device_manager.gpu
        self.device_manager.to_device(self.encoder, self.decoder)

    def fit(self, train, validate, epochs=32, start_epoch=1, data_depth=None):
        """Train the model."""
        if self.data_depth is None:
            self.data_depth = data_depth

        if self.encoder_decoder_optimizer is None:
            self.encoder_decoder_optimizer = self.training_manager.get_optimizer()
            self.training_manager.optimizer = self.encoder_decoder_optimizer

        if self.history_manager:
            self.history = self.history_manager.history

        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print(f'\n{"="*60}')
                print(f'Epoch {epoch}/{end_epoch - 1}')
                print(f'{"="*60}')

            self.training_manager.fit_coders(train, metrics)
            self.training_manager.validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch

            if self.verbose and self.history_manager:
                self.history_manager.print_metrics(self.fit_metrics)

            if self.log_dir:
                if self.history_manager:
                    self.history_manager.append_and_save(self.fit_metrics)
                    self.history = self.history_manager.history

                if epoch == start_epoch or epoch % 5 == 0 or epoch == end_epoch - 1:
                    save_name = '{}.rsbpp-{:03f}.p'.format(epoch, self.fit_metrics['val.rsbpp'])
                    self.save(os.path.join(self.log_dir, save_name))
                    sample_generator = SampleGenerator(
                        self.encoder, self.payload_generator, self.device
                    )
                    sample_generator.generate_samples(
                        self.samples_path, epoch, "Hello, SteganoGAN!", self.data_depth
                    )

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()

            gc.collect()

    def encode(self, cover, output, text):
        """Encode text message into cover image."""
        self.encoder_manager.encode(cover, output, text)

    def decode(self, image):
        """Decode hidden message from image."""
        return self.decoder_manager.decode(image)

    def save(self, path):
        """Save model to file."""
        torch.save(self, path)
        if self.verbose:
            print(f'Model saved to {path}')

    @classmethod
    def load(cls, path, gpu=True, verbose=False, log_dir=None):
        """Load model from file."""
        return ModelLoader.load(path, gpu, verbose, log_dir)
