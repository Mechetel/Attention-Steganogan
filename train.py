#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SteganoGAN Training Script
Train a steganography model with tqdm progress bars and automatic metrics logging.
"""

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import json
from time import time
import torch

from steganogan.steganogan import SteganoGAN
from steganogan.models.decoders import DenseDecoder, BasicDecoder
from steganogan.models.encoders import DenseEncoder, BasicEncoder, ResidualEncoder #, AttentionEncoder
from steganogan.data_loader import DataLoader


def main():
    """Main training function."""
    # Training configuration
    CONFIG = {
        'gpu': True,
        'data_depth': 1,
        'encoder': 'dense',  # Options: 'basic', 'residual', 'dense', 'attention'
        'decoder': 'dense',  # Options: 'basic', 'dense'
        'epochs': 1,
        'batch_size': 4,  # Reduced to 1 for attention encoder memory requirements
        'num_workers': 8, # Set to 0 for macOS/MPS compatibility # 8
        'dataset': 'div2k',
        'training_type': 'panet_dense'
    }

    # Set random seed for reproducibility
    torch.manual_seed(42)

    print("="*60)
    print("SteganoGAN Training")
    print("="*60)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()

    # Select encoder and decoder based on config
    encoder_map = {
        'basic': BasicEncoder,
        'residual': ResidualEncoder,
        'dense': DenseEncoder
        # 'attention': AttentionEncoder
    }

    decoder_map = {
        'basic': BasicDecoder,
        'dense': DenseDecoder
    }

    encoder_class = encoder_map[CONFIG['encoder']]
    decoder_class = decoder_map[CONFIG['decoder']]

    # Create data loaders
    print("Loading datasets...")
    train = DataLoader(
        os.path.expanduser('~/Attention-Steganogan/data/div2k/train'),
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        shuffle=True
    )

    validation = DataLoader(
        os.path.expanduser('~/Attention-Steganogan/data/div2k/val'),
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        shuffle=False
    )

    print(f"Train dataset size: {len(train.dataset)}")
    print(f"Validation dataset size: {len(validation.dataset)}")

    # Create output directory
    timestamp = str(int(time()))
    log_dir = os.path.join('models', CONFIG['training_type'], timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"\nLogs and checkpoints will be saved to: {log_dir}")

    # Initialize SteganoGAN model
    print(f"\nInitializing model with {CONFIG['encoder']} encoder and {CONFIG['decoder']} decoder...")
    steganogan = SteganoGAN(
        data_depth=CONFIG['data_depth'],
        encoder=encoder_class,
        decoder=decoder_class,
        gpu=CONFIG['gpu'],
        verbose=True,
        log_dir=log_dir
    )

    # Train the model
    print("\nStarting training...")
    steganogan.fit(train, validation, epochs=CONFIG['epochs'])

    # Save final model
    final_weights_path = os.path.join(log_dir, "weights.steg")
    steganogan.save(final_weights_path)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final model saved to: {final_weights_path}")
    print(f"Metrics saved to: {os.path.join(log_dir, 'metrics.log')}")

    # Display final metrics
    if steganogan.fit_metrics:
        print("\nFinal Epoch Metrics:")
        for key, value in steganogan.fit_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
