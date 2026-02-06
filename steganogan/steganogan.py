# -*- coding: utf-8 -*-
import gc
import inspect
import os
import torch

from steganogan.managers.decoder_manager import DecoderManager
from steganogan.managers.encoder_manager import EncoderManager
from steganogan.managers.history_manager import HistoryManager
from steganogan.managers.training_manager import TrainingManager
from steganogan.managers.device_manager import DeviceManager
from steganogan.generators.sample_generator import SampleGenerator
from steganogan.model_loader import ModelLoader
from steganogan.generators.payload_generator import PayloadGenerator

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
