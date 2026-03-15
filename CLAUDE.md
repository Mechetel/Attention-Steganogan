# Attention-Steganogan Project Context

## Project Overview
This is a PhD dissertation project implementing Steganogan with attention mechanisms for image steganography. The goal is to develop a neural network architecture that can hide images within images with improved capacity and security using attention mechanisms.

## Core Research References

### 1. **SteganoGAN.pdf**
Foundation paper on SteganoGAN - adversarial approach to image steganography using GANs. Covers:
- GAN-based steganography framework
- Generator and discriminator architecture
- Adversarial training for natural-looking stego images

**Location:** `/Users/dmitryhoma/Desktop/steganogan attention articles/SteganoGAN.pdf`

### 2. **Steganogan Unet + SA.pdf**
Hybrid architecture combining UNet with Self-Attention (SA) mechanisms. Key contributions:
- UNet encoder-decoder for improved image processing
- Self-attention modules for better feature learning
- Enhanced capacity while maintaining invisibility

**Location:** `/Users/dmitryhoma/Desktop/steganogan attention articles/Steganogan Unet + SA.pdf`

### 3. **HCISNet_Higher-capacity_invisible_image_steganogra.pdf**
Higher-Capacity Invisible Steganography Network (HCISNet). Focuses on:
- Increasing capacity for hidden data while maintaining imperceptibility
- Advanced network architectures for steganography
- Attention mechanisms for capacity enhancement

**Location:** `/Users/dmitryhoma/Desktop/steganogan attention articles/HCISNet_Higher-capacity_invisible_image_steganogra.pdf`

## Key Implementation Areas

### Model Architecture
- `steganogan/` - Main implementation directory
- Focus: GAN-based generator and discriminator with attention mechanisms
- UNet backbone with self-attention blocks

### Training & Testing
- `train.py` / `train.ipynb` - Model training pipeline
- `test.py` / `test.ipynb` - Evaluation and testing
- Loss functions: image quality (MSE), adversarial loss, capacity metrics

### Data
- `data/` - Dataset directory for input/secret images
- Input/output samples in project root (input.png, output.png)

## Important Notes for Development

1. **Attention Mechanisms** - Core innovation: use self-attention in encoder/decoder paths
2. **Loss Balancing** - Image quality loss weight currently set to 100 for MSE
3. **Capacity vs. Quality Trade-off** - Balance between hidden data capacity and stego image invisibility
4. **No-grad Operations** - Coding scores computed with `no_grad()` for efficiency
5. **Model Checkpoints** - Trained dense model weights available in `models/`

## Useful Commands

```bash
# Train model
python train.py

# Test/evaluate
python test.py

# Interactive notebook
jupyter notebook train.ipynb
```

---

*These research papers form the theoretical foundation for the Attention-Steganogan implementation and should be referenced when making architectural decisions.*
