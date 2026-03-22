# Attention-SteganoGAN — PhD Dissertation Project

## Project Overview

Image steganography system based on deep learning. Embeds secret binary messages into cover images producing visually imperceptible stego images.

## Architecture Variants

### Encoders
- **BasicEncoder / ResidualEncoder / DenseEncoder** — Original SteganoGAN architectures (Zhang et al., 2019)
- **EdgeGuidedDualStreamUNetEncoder** — Dual-stream U-Net with MSMA attention + InceptionDMK + ConvGRU iterative refinement (Ji, Zhang, Lv — Applied Sciences 2025)
- **EdgeAwareDenseASPPEncoder** — Novel: DenseASPP backbone + MSMA + learned EdgeNet + edge-masked ConvGRU refinement. Concentrates data embedding in edge regions.
- **DepthAgnosticEncoder** — 5ch input (3 cover + 1 bit-plane + 1 normalised index). Folds D into batch inside `forward()` — one GPU call, no Python loop. `sample_depth()` draws random D∈[1,data_depth] each training batch so one model works for all depths at inference. Set inference depth via `model.set_depth(d)`.

### Decoders
- **BasicDecoder / DenseDecoder** — Original SteganoGAN decoders
- **EdgeAwareDenseDecoder** — Edge-aware decoder with lightweight DenseASPP + MSMA attention
- **DepthAgnosticDecoder** — 4ch input (3 stego + 1 normalised index). Folds D into batch inside `forward()` — one GPU call, no Python loop. Reads D from `self.data_depth` (synced by Trainer per batch; set inference depth via `model.set_depth(d)`). Output: (N, D, H, W) logits.

### Critics
- **BasicCritic** — Original WGAN critic with weight clipping
- **MultiScaleEdgeAwareCritic** — Multi-scale (3 scales) with spectral normalisation and Sobel edge input

## Reference Papers
1. **SteganoGAN** — Zhang et al. "SteganoGAN: High Capacity Image Steganography with GANs" (2019)
2. **HCISNet** — "Higher-Capacity Invisible Image Steganographic Network" — Enhanced DenseASPP encoder
3. **Edge-Guided U-Net** — Ji, Zhang, Lv "Edge-Guided Dual-Stream U-Net for Secure Image Steganography" (Applied Sciences 2025) — MSMA + InceptionDMK + ConvGRU
4. **XuNet** — Xu et al. "Structural Design of Convolutional Neural Networks for Steganalysis" (IEEE SPL 2016)
5. **YeNet** — Ye et al. "Deep Learning Hierarchical Representations for Image Steganalysis" (IEEE TIFS 2017)
6. **SRNet** — Boroumand et al. "Deep Residual Network for Steganalysis of Digital Images" (IEEE TIFS 2019) — 2×Type1 + 5×Type2 + 4×Type3 + Type4(256→512) + FC(512→2)
7. **YedroudjNet** — Yedroudj et al. "Yedroudj-Net: An Efficient CNN for Spatial Steganalysis" (ICASSP 2018)
8. **ZhuNet** — Zhu et al. deep steganalysis network with SRM preprocessing (2020)
9. **SiaStegNet** — You, Zhang & Zhao "A Siamese CNN for Image Steganalysis" (IEEE TIFS 2021)

## Key Design Decisions
- Images normalised to [-1, 1] range throughout
- Iterative encoders return `List[Tensor]` during training, single `Tensor` during inference
- WGAN training: critic trains 5 steps per encoder step
- EdgeNet trained end-to-end (no pretrained edge detection) with Sobel regularisation (λ=0.01)
- Edge mask uses epsilon floor (0.05) to prevent dead gradients in flat regions

## Training (Steganography)
```bash
python train.py
```
Configure architecture choice and hyperparameters in `CONFIG` dict at top of `train.py`.

## Training (Steganalysis)
```bash
python steganalyzers/train.py
python steganalyzers/test.py
```
Configure network choice and hyperparameters in `CONFIG` dict at top of each file.

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

steganalyzers/
├── base.py                        # BaseSteganalyzer ABC
├── kernels/srm.py                 # 30 SRM 5×5 high-pass filters
├── models/
│   ├── xunet.py                   # XuNet (Xu et al., 2016)
│   ├── yenet.py                   # YeNet (Ye et al., 2017) — trainable SRM + TLU
│   ├── srnet.py                   # SRNet (Boroumand et al., 2019) — Type I/II/III/IV
│   ├── yedroudjnet.py             # YedroudjNet (Yedroudj et al., 2018) — fixed SRM
│   ├── zhunet.py                  # ZhuNet (Zhu et al., 2020) — deeper SRNet-style
│   └── siastegnet.py              # SiaStegNet (You et al., 2021) — Siamese CNN
│                                  #   forward(x) → (N,2) single-image
│                                  #   siamese_forward(x1,x2) → (N,1) pair mode
├── data/
│   ├── alaska2.py                 # Alaska2Dataset (Cover/JMiPOD/JUNIWARD/UERD)
│   └── dataloader.py              # DataLoaderFactory (balanced sampler)
├── training/
│   ├── trainer.py                 # Trainer (AMP, grad clip, callback system)
│   ├── metrics.py                 # accuracy, balanced_acc, AUC-ROC, TPR@FPR0.1, F1
│   └── callbacks.py               # MetricsLogger, CheckpointSaver, EarlyStopping, LRMonitor
├── utils/checkpoint.py            # Checkpoint save/load
├── train.py                       # Training entry point
└── test.py                        # Evaluation entry point
```

## RGB Adaptation (steganalyzers)
All networks accept 3-channel RGB input. SRM-based networks use grouped convolution
`Conv2d(3, 30×3, 5, groups=3)` so each channel is filtered independently by the 30
SRM kernels, then a 1×1 conv projects back to the target channel width.

## ALASKA2 Dataset Layout
```
~/Projects/datasets/alaska2-image-steganalysis/
├── Cover/      *.jpg   label 0  (75 000 images, 512×512)
├── JMiPOD/     *.jpg   label 1  (75 000 images)
├── JUNIWARD/   *.jpg   label 1  (75 000 images)
├── UERD/       *.jpg   label 1  (75 000 images)
└── Test/       *.jpg   unlabeled (5 000 Kaggle competition images)
```
