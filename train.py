#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SteganoGAN Training Script
────────────────────────────────────────────────────────────────────────────────
Supports all encoder variants including the new
  EdgeGuidedDualStreamUNetEncoder  (Ji, Zhang, Lv – Applied Sciences 2025).

Recommended settings to reproduce the paper results:
  encoder        : 'edge_unet'
  decoder        : 'dense'
  critic         : True
  data_depth     : 1 / 2 / 3 / 4     (bits per pixel)
  batch_size     : 2  (paper used batch_size=2 on NVIDIA V100)
  epochs         : 100
  T              : 10   (GRU iteration steps)
  eta            : 1.0  (perturbation step size)
  gamma          : 0.8  (iterative loss decay)
  alpha          : 1.0  (image quality loss weight)
  image_size     : 360  (images resized to 360×360×3)
"""

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import json
from time import time
import torch

from steganogan.steganogan import SteganoGAN
from steganogan.models.decoders  import DenseDecoder, BasicDecoder
from steganogan.models.encoders  import (
    DenseEncoder,
    BasicEncoder,
    ResidualEncoder,
    EdgeGuidedDualStreamUNetEncoder,
)
from steganogan.models.critics   import BasicCritic
from steganogan.data_loader      import DataLoader


def main():
    """Main training function."""

    # ── Configuration ─────────────────────────────────────────────────────
    CONFIG = {
        'gpu'          : True,
        'data_depth'   : 1,          # bits per pixel; try 1, 2, 3, 4

        # Encoder choice:
        #   'basic'     – simple 3-layer CNN
        #   'residual'  – BasicEncoder + residual connection
        #   'dense'     – DenseNet-style (SteganoGAN original)
        #   'edge_unet' – Edge-Guided Dual-Stream U-Net (Ji et al. 2025)
        'encoder'      : 'edge_unet',
        'decoder'      : 'dense',    # 'basic' | 'dense'
        'critic'       : True,

        'epochs'       : 100,
        'batch_size'   : 2,          # paper used 2 on V100
        'num_workers'  : 4,

        'dataset'      : 'div2k',
        'training_type': 'edge_unet_dense',

        # ── EdgeGuidedDualStreamUNetEncoder hyper-params ────────────────
        'T'            : 10,         # GRU iteration steps
        'eta'          : 1.0,        # perturbation step size
        'gamma'        : 0.8,        # iterative loss decay factor
        'alpha'        : 1.0,        # image quality loss weight (α)
        'sobel_alpha'  : 1.0,        # edge enhancement strength
        'hidden_ch'    : 32,         # ConvGRU hidden channels
    }

    torch.manual_seed(42)

    print('=' * 60)
    print('SteganoGAN Training – Edge-Guided Dual-Stream U-Net')
    print('=' * 60)
    print('\nConfiguration:')
    for k, v in CONFIG.items():
        print(f'  {k}: {v}')
    print()

    # ── Build encoder ─────────────────────────────────────────────────────
    encoder_map = {
        'basic'    : lambda d: BasicEncoder(d),
        'residual' : lambda d: ResidualEncoder(d),
        'dense'    : lambda d: DenseEncoder(d),
        'edge_unet': lambda d: EdgeGuidedDualStreamUNetEncoder(
            data_depth  = d,
            T           = CONFIG['T'],
            eta         = CONFIG['eta'],
            gamma       = CONFIG['gamma'],
            alpha       = CONFIG['alpha'],
            sobel_alpha = CONFIG['sobel_alpha'],
            hidden_ch   = CONFIG['hidden_ch'],
        ),
    }

    decoder_map = {
        'basic': BasicDecoder,
        'dense': DenseDecoder,
    }

    encoder_class  = encoder_map[CONFIG['encoder']]
    decoder_class  = decoder_map[CONFIG['decoder']]
    critic_class   = BasicCritic if CONFIG['critic'] else None

    # Instantiate encoder (edge_unet is a factory lambda, others are classes)
    encoder_instance = encoder_class(CONFIG['data_depth'])

    # ── Data loaders ──────────────────────────────────────────────────────
    print('Loading datasets...')
    data_root = os.path.expanduser('~/Attention-Steganogan/data')

    train = DataLoader(
        os.path.join(data_root, CONFIG['dataset'], 'train'),
        batch_size  = CONFIG['batch_size'],
        num_workers = CONFIG['num_workers'],
        shuffle     = True,
    )
    validation = DataLoader(
        os.path.join(data_root, CONFIG['dataset'], 'val'),
        batch_size  = CONFIG['batch_size'],
        num_workers = CONFIG['num_workers'],
        shuffle     = False,
    )

    print(f'  Train size      : {len(train.dataset)}')
    print(f'  Validation size : {len(validation.dataset)}')

    # ── Output directory ──────────────────────────────────────────────────
    timestamp = str(int(time()))
    log_dir   = os.path.join('models', CONFIG['training_type'], timestamp)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)

    print(f'\nLogs → {log_dir}')

    # ── Initialise SteganoGAN ─────────────────────────────────────────────
    print(f'\nInitialising model …')
    if CONFIG['critic']:
        print('  Adversarial training ENABLED')

    steganogan = SteganoGAN(
        data_depth = CONFIG['data_depth'],
        encoder    = encoder_instance,      # already instantiated
        decoder    = decoder_class,
        critic     = critic_class,
        gpu        = CONFIG['gpu'],
        verbose    = True,
        log_dir    = log_dir,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print('\nStarting training …\n')
    steganogan.fit(train, validation, epochs=CONFIG['epochs'])

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(log_dir, 'weights.steg')
    steganogan.save(final_path)

    print('\n' + '=' * 60)
    print('Training complete.')
    print(f'  Model   : {final_path}')
    print(f'  Metrics : {os.path.join(log_dir, "metrics.log")}')
    print('=' * 60)

    if steganogan.fit_metrics:
        print('\nFinal epoch metrics:')
        for k, v in steganogan.fit_metrics.items():
            print(f'  {k}: {v:.6f}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
