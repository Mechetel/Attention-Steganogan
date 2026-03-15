# Attention-SteganoGAN — PhD Dissertation Project

## Project Overview

Image steganography system based on deep learning. Embeds secret binary messages into cover images producing visually imperceptible stego images.

## Architecture Variants

### Encoders
- **BasicEncoder / ResidualEncoder / DenseEncoder** — Original SteganoGAN architectures (Zhang et al., 2019)
- **EdgeGuidedDualStreamUNetEncoder** — Dual-stream U-Net with MSMA attention + InceptionDMK + ConvGRU iterative refinement (Ji, Zhang, Lv — Applied Sciences 2025)
- **EdgeAwareDenseASPPEncoder** — Novel: DenseASPP backbone + MSMA + learned EdgeNet + edge-masked ConvGRU refinement. Concentrates data embedding in edge regions.

### Decoders
- **BasicDecoder / DenseDecoder** — Original SteganoGAN decoders
- **EdgeAwareDenseDecoder** — Edge-aware decoder with lightweight DenseASPP + MSMA attention

### Critics
- **BasicCritic** — Original WGAN critic with weight clipping
- **MultiScaleEdgeAwareCritic** — Multi-scale (3 scales) with spectral normalisation and Sobel edge input

## Reference Papers
1. **SteganoGAN** — Zhang et al. "SteganoGAN: High Capacity Image Steganography with GANs" (2019)
2. **HCISNet** — "Higher-Capacity Invisible Image Steganographic Network" — Enhanced DenseASPP encoder
3. **Edge-Guided U-Net** — Ji, Zhang, Lv "Edge-Guided Dual-Stream U-Net for Secure Image Steganography" (Applied Sciences 2025) — MSMA + InceptionDMK + ConvGRU

## Key Design Decisions
- Images normalised to [-1, 1] range throughout
- Iterative encoders return `List[Tensor]` during training, single `Tensor` during inference
- WGAN training: critic trains 5 steps per encoder step
- EdgeNet trained end-to-end (no pretrained edge detection) with Sobel regularisation (λ=0.01)
- Edge mask uses epsilon floor (0.05) to prevent dead gradients in flat regions

## Training
```bash
python train.py
```
Configure architecture choice and hyperparameters in `CONFIG` dict at top of `train.py`.

## Project Structure
```
steganogan/
├── models/
│   ├── base.py                    # BaseEncoder, BaseDecoder, BaseCritic ABCs
│   ├── encoders/
│   │   ├── basic.py               # BasicEncoder, ResidualEncoder, DenseEncoder
│   │   ├── edge_unet/             # EdgeGuidedDualStreamUNetEncoder
│   │   └── edge_aspp/             # EdgeAwareDenseASPPEncoder (novel)
│   ├── decoders/decoders.py       # All decoder variants
│   └── critics/critics.py        # All critic variants
├── training/
│   ├── trainer.py                 # Training loop
│   ├── losses.py                  # Loss functions (incl. VGGPerceptualLoss)
│   └── metrics.py                 # PSNR, SSIM, RSBPP metrics
├── data/                          # Dataset, DataLoader, transforms
├── inference/                     # Encode/decode services
└── utils/                         # Payload, checkpoints, visualisation
```