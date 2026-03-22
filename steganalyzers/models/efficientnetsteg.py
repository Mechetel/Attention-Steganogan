# -*- coding: utf-8 -*-
"""
EfficientNetSteg — EfficientNet-B0 fine-tuned for image steganalysis.

Architecture
------------
SRM pre-processing  : grouped Conv2d(3, 90, 5, groups=3) → abs → 1×1 proj → 3 ch
                      Highlights noise residuals before the main backbone.
EfficientNet-B0     : pretrained on ImageNet1K, backbone features (1 280-dim).
Classifier head     : Dropout → Linear(1 280 → 512) → ReLU → Dropout →
                      Linear(512 → num_classes)

The SRM layer is always trained from scratch.  The EfficientNet backbone is
loaded with ImageNet weights.  Layers up to (but not including) ``unfreeze_from``
can be frozen for the initial warm-up phase and later unlocked.

References
----------
Tan, M. & Le, Q. V., "EfficientNet: Rethinking Model Scaling for Convolutional
Neural Networks", ICML 2019.

Boroumand et al., "Deep Residual Network for Steganalysis of Digital Images",
IEEE TIFS 2019  (SRM preprocessing concept).
"""

import torch
import torch.nn as nn

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    _WEIGHTS_ENUM = True
except ImportError:
    from torchvision.models import efficientnet_b0
    _WEIGHTS_ENUM = False

from ..base import BaseSteganalyzer
from ..kernels import get_srm_kernels


# ── SRM pre-processor ──────────────────────────────────────────────────────────

class _SRMPreprocessor(nn.Module):
    """
    Channel-independent SRM high-pass filter followed by a 1×1 projection.

    Each input channel is convolved independently with the 30 SRM filters
    (grouped convolution), then absolute value is taken, and a 1×1 conv
    projects the 90-channel map back to ``out_channels`` (default 3) so the
    EfficientNet backbone receives a standard 3-channel tensor.

    Parameters
    ----------
    in_channels  : number of input image channels (3 for RGB)
    out_channels : channels passed to the backbone (default 3)
    srm_trainable: whether to fine-tune the SRM kernel weights
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        srm_trainable: bool = True,
    ) -> None:
        super().__init__()
        kernels = torch.from_numpy(get_srm_kernels())    # (30, 1, 5, 5)
        # Replicate for each input channel → (30*in_channels, 1, 5, 5)
        kernels = kernels.repeat(in_channels, 1, 1, 1)

        self.srm = nn.Conv2d(
            in_channels, 30 * in_channels,
            kernel_size=5, padding=2, groups=in_channels, bias=False,
        )
        with torch.no_grad():
            self.srm.weight.copy_(kernels)

        if not srm_trainable:
            self.srm.weight.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Conv2d(30 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.srm(x).abs()
        return self.proj(x)


# ── Main model ─────────────────────────────────────────────────────────────────

class EfficientNetSteg(BaseSteganalyzer):
    """
    EfficientNet-B0 steganalyzer with SRM preprocessing.

    Parameters
    ----------
    in_channels    : input image channels (default 3)
    num_classes    : output logits (default 2: cover / stego)
    srm_trainable  : fine-tune SRM preprocessing filters (default True)
    freeze_backbone: freeze EfficientNet backbone weights during training.
                     Useful for a short warm-up of the head; call
                     :meth:`unfreeze_backbone` to unlock all layers later.
    dropout        : dropout probability in the classifier head (default 0.4)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        srm_trainable: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.4,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)

        # ── SRM pre-processor ────────────────────────────────────────────────
        self.preprocessor = _SRMPreprocessor(
            in_channels=in_channels,
            out_channels=3,          # backbone always sees 3 channels
            srm_trainable=srm_trainable,
        )

        # ── EfficientNet-B0 backbone (pretrained) ────────────────────────────
        if _WEIGHTS_ENUM:
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b0(pretrained=True)

        # Keep only the feature extractor; drop the original classifier
        self.features = backbone.features       # outputs (N, 1280, H/32, W/32)
        self.avgpool  = backbone.avgpool        # AdaptiveAvgPool2d → (N, 1280, 1, 1)

        backbone_out = 1280

        # ── Custom steganalysis head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        self._init_head()

        if freeze_backbone:
            self.freeze_backbone()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_head(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Backbone freeze / unfreeze helpers ────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all EfficientNet backbone parameters."""
        for p in self.features.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all EfficientNet backbone parameters."""
        for p in self.features.parameters():
            p.requires_grad_(True)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessor(x)              # SRM noise residuals → (N, 3, H, W)
        x = self.features(x)                  # (N, 1280, H/32, W/32)
        x = self.avgpool(x)                   # (N, 1280, 1, 1)
        x = torch.flatten(x, 1)              # (N, 1280)
        return self.classifier(x)             # (N, num_classes)
