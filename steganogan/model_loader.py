import os
from typing import Optional
import torch
from .managers.device_manager import DeviceManager


class ModelLoader:
    """Handles model loading and initialization."""

    @staticmethod
    def load(
        path: str,
        gpu: bool = True,
        verbose: bool = False,
        log_dir: Optional[str] = None,
    ) -> object:
        """
        Load a SteganoGAN model from a saved checkpoint file.

        Parameters
        ----------
        path    : path to the saved .steg / .p checkpoint
        gpu     : if True, move model to CUDA / MPS when available
        verbose : print status messages when True
        log_dir : optional directory for saving logs and samples

        Returns
        -------
        Loaded and device-ready SteganoGAN instance.
        """
        if path is None:
            raise ValueError('Please provide a path to pretrained model.')

        # Mirror DeviceManager logic for consistent device selection
        if gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        if verbose:
            print(f'Loading model to device: {device}')

        steganogan = torch.load(path, map_location=device, weights_only=False)
        steganogan.verbose = verbose

        steganogan.encoder_decoder_optimizer = None
        steganogan.fit_metrics               = None
        steganogan.history                   = []

        steganogan.log_dir = log_dir
        if log_dir:
            os.makedirs(steganogan.log_dir, exist_ok=True)
            steganogan.samples_path = os.path.join(steganogan.log_dir, 'samples')
            os.makedirs(steganogan.samples_path, exist_ok=True)

        # Upgrade legacy sub-modules
        steganogan.encoder.upgrade_legacy()
        steganogan.decoder.upgrade_legacy()
        if hasattr(steganogan, 'critic') and steganogan.critic is not None:
            steganogan.critic.upgrade_legacy()

        # Reinitialise device manager so .device / .gpu stay consistent
        steganogan.device_manager = DeviceManager(gpu=gpu, verbose=verbose)
        steganogan.device = steganogan.device_manager.device
        steganogan.gpu    = steganogan.device_manager.gpu

        models = [steganogan.encoder, steganogan.decoder]
        if hasattr(steganogan, 'critic') and steganogan.critic is not None:
            models.append(steganogan.critic)
        steganogan.device_manager.to_device(*models)

        if verbose:
            print(f'Model loaded from {path}')

        return steganogan
