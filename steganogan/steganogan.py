# -*- coding: utf-8 -*-
import gc
import inspect
import os
from typing import Any, Dict, List, Optional, Type, Union
import torch

from .managers import (
    DecoderManager,
    EncoderManager,
    HistoryManager,
    TrainingManager,
    DeviceManager,
)
from .generators import SampleGenerator, PayloadGenerator
from .model_loader import ModelLoader

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train'
)

METRIC_FIELDS: List[str] = [
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

CRITIC_METRIC_FIELDS: List[str] = [
    'train.cover_score',
    'train.generated_score',
    'val.cover_score',
    'val.generated_score',
]


class SteganoGAN(object):
    """Main SteganoGAN model class."""

    def __init__(
        self,
        data_depth: int,
        encoder: Union[torch.nn.Module, Type[torch.nn.Module]],
        decoder: Union[torch.nn.Module, Type[torch.nn.Module]],
        critic: Optional[Union[torch.nn.Module, Type[torch.nn.Module]]] = None,
        gpu: bool = True,
        verbose: bool = False,
        log_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.verbose:    bool = verbose
        self.data_depth: int  = data_depth
        kwargs['data_depth']  = data_depth

        self.encoder: torch.nn.Module = self._get_instance(encoder, kwargs)
        self.decoder: torch.nn.Module = self._get_instance(decoder, kwargs)

        self.critic: Optional[torch.nn.Module] = (
            self._get_instance(critic, kwargs) if critic is not None else None
        )

        self.device_manager: DeviceManager = DeviceManager(gpu=gpu, verbose=verbose)
        self.device: torch.device          = self.device_manager.device
        self.gpu:    bool                  = self.device_manager.gpu

        models_to_device = [self.encoder, self.decoder]
        if self.critic is not None:
            models_to_device.append(self.critic)
        self.device_manager.to_device(*models_to_device)

        self.payload_generator: PayloadGenerator = PayloadGenerator(self.device)
        self.training_manager:  TrainingManager  = TrainingManager(
            self.encoder, self.decoder, self.data_depth, self.device, verbose,
            critic=self.critic,
        )

        self.encoder_decoder_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer:          Optional[torch.optim.Optimizer] = None
        self.fit_metrics:               Optional[Dict[str, Any]]        = None
        self.history:                   List[Dict[str, Any]]            = []

        self.log_dir: Optional[str] = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path: str = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)
            self.history_manager: Optional[HistoryManager] = HistoryManager(log_dir)
        else:
            self.history_manager = None

        self.encoder_manager: EncoderManager = EncoderManager(
            self.encoder, self.payload_generator, self.data_depth, self.device, verbose
        )
        self.decoder_manager: DecoderManager = DecoderManager(
            self.decoder, self.device, verbose
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_instance(
        self,
        class_or_instance: Union[torch.nn.Module, Type[torch.nn.Module]],
        kwargs: Dict[str, Any],
    ) -> torch.nn.Module:
        """Return `class_or_instance` directly if already instantiated,
        otherwise instantiate it with the relevant subset of `kwargs`."""
        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec if arg in kwargs}
        return class_or_instance(**init_args)

    # ── Device management ─────────────────────────────────────────────────────

    def set_device(self) -> None:
        """Re-detect and apply the best available compute device."""
        self.device_manager.set_device()
        self.device = self.device_manager.device
        self.gpu    = self.device_manager.gpu
        models = [self.encoder, self.decoder]
        if self.critic is not None:
            models.append(self.critic)
        self.device_manager.to_device(*models)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train: torch.utils.data.DataLoader,
        validate: torch.utils.data.DataLoader,
        epochs: int = 32,
        start_epoch: int = 1,
        data_depth: Optional[int] = None,
    ) -> None:
        """
        Train the encoder, decoder, and (optionally) the critic.

        Parameters
        ----------
        train       : training DataLoader
        validate    : validation DataLoader
        epochs      : number of epochs to run
        start_epoch : epoch index to start from (useful when resuming)
        data_depth  : override self.data_depth if provided
        """
        if self.data_depth is None:
            self.data_depth = data_depth

        if self.encoder_decoder_optimizer is None:
            self.encoder_decoder_optimizer = self.training_manager.get_optimizer()
            self.training_manager.optimizer = self.encoder_decoder_optimizer

        if self.critic is not None and self.critic_optimizer is None:
            self.critic_optimizer = self.training_manager.get_critic_optimizer()
            self.training_manager.critic_optimizer = self.critic_optimizer

        if self.history_manager:
            self.history = self.history_manager.history

        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            all_fields = METRIC_FIELDS + (
                CRITIC_METRIC_FIELDS if self.critic is not None else []
            )
            metrics: Dict[str, List[float]] = {f: [] for f in all_fields}

            if self.verbose:
                print(f'\n{"=" * 60}')
                print(f'Epoch {epoch}/{end_epoch - 1}')
                print(f'{"=" * 60}')

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
                    save_name = '{}.rsbpp-{:03f}.p'.format(
                        epoch, self.fit_metrics['val.rsbpp']
                    )
                    self.save(os.path.join(self.log_dir, save_name))
                    SampleGenerator(
                        self.encoder, self.payload_generator, self.device
                    ).generate_samples(
                        self.samples_path, epoch,
                        'Hello, SteganoGAN!', self.data_depth,
                    )

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()

            gc.collect()

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, cover: str, output: str, text: str) -> None:
        """Embed `text` into the cover image at `cover` and write to `output`."""
        self.encoder_manager.encode(cover, output, text)

    def decode(self, image: str) -> str:
        """Extract and return the hidden message from the stego image at `image`."""
        return self.decoder_manager.decode(image)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the full model to `path`."""
        torch.save(self, path)
        if self.verbose:
            print(f'Model saved to {path}')

    @classmethod
    def load(cls, path: str, gpu: bool,
             verbose: bool = False,
             log_dir: Optional[str] = None) -> 'SteganoGAN':
        """Load and return a SteganoGAN model from a checkpoint file."""
        return ModelLoader.load(path, gpu, verbose, log_dir)
