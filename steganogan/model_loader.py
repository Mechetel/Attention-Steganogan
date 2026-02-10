import os
import torch
from .managers.device_manager import DeviceManager

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
