# -*- coding: utf-8 -*-
"""Model serialisation: save and load SteganoGAN checkpoints."""

import os
from typing import Optional, TYPE_CHECKING

import torch

from .device import DeviceManager

if TYPE_CHECKING:
    from ..engine import SteganoGAN


class ModelCheckpoint:
    """
    Handles saving and loading of SteganoGAN checkpoints.
    All methods are static; this class is a pure-utility namespace.
    """

    @staticmethod
    def save(model: "SteganoGAN", path: str, verbose: bool = False) -> None:
        """Serialise *model* to *path* using :func:`torch.save`."""
        torch.save(model, path)
        if verbose:
            print(f"Checkpoint saved → {path}")

    @staticmethod
    def load(
        path:    str,
        gpu:     bool = True,
        verbose: bool = False,
        log_dir: Optional[str] = None,
    ) -> "SteganoGAN":
        """
        Deserialise a SteganoGAN checkpoint and prepare it for inference or
        resumed training.

        Parameters
        ----------
        path    : path to the ``.steg`` / ``.p`` file
        gpu     : move model to GPU when True and a GPU is available
        verbose : print status messages
        log_dir : optional directory for logs / samples
        """
        if path is None:
            raise ValueError("Checkpoint path must not be None.")

        device_mgr = DeviceManager(gpu=gpu, verbose=verbose)
        if verbose:
            print(f"Loading checkpoint: {path} → {device_mgr.device}")

        model = torch.load(path, map_location=device_mgr.device, weights_only=False)
        model.verbose = verbose

        # Reset transient optimiser / metric state
        model.encoder_decoder_optimizer = None
        model.fit_metrics               = None
        model.history                   = []

        # Optional log dir
        model.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            model.samples_path = os.path.join(log_dir, "samples")
            os.makedirs(model.samples_path, exist_ok=True)

        # Upgrade any legacy sub-modules
        for attr in ("encoder", "decoder", "critic"):
            sub = getattr(model, attr, None)
            if sub is not None and hasattr(sub, "upgrade_legacy"):
                sub.upgrade_legacy()

        # Sync device manager
        model.device_manager = device_mgr
        model.device         = device_mgr.device
        model.gpu            = device_mgr.gpu

        modules = [model.encoder, model.decoder]
        if getattr(model, "critic", None) is not None:
            modules.append(model.critic)
        device_mgr.to_device(*modules)

        # Sync device on any serialised sub-services that cache it
        for svc in ("_payload_factory", "_encoder_svc", "_decoder_svc", "_trainer"):
            obj = getattr(model, svc, None)
            if obj is not None and hasattr(obj, "device"):
                obj.device = device_mgr.device

        if verbose:
            print("Checkpoint loaded successfully.")
        return model
