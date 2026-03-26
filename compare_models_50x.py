"""
Loads Dense and Edge-ASPP SteganoGAN checkpoints, runs inference via EncoderService,
and produces a side-by-side comparison:
  Original | Dense stego | Edge-ASPP stego | Dense residual (L) | Edge-ASPP residual (L)

Residuals are computed at full resolution before any downscaling.
"""
import sys, os, tempfile, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda.amp")

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from steganogan.engine import SteganoGAN

BASE = os.path.dirname(__file__)

DENSE_CKPT    = os.path.join(BASE, "models/div2k/dense_dense/1773340283/32.rsbpp-0.965443.p")
EDGEUNET_CKPT = os.path.join(BASE, "models/div2k/edge_unet_dense/1773386504/32.rsbpp-0.969229.p")
EDGEASPP_CKPT = os.path.join(BASE, "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/32.rsbpp-0.971016.p")
COVER_PATH    = os.path.join(BASE, "data/callback_images/0015.png")
OUT_PATH      = os.path.join(BASE, "comparison_50x.png")
MESSAGE       = "Hello, SteganoGAN!"
DISPLAY_SIZE  = 360
AMPLIFY       = 50


def encode_via_service(model: SteganoGAN, cover_path: str, message: str) -> np.ndarray:
    """Use EncoderService to encode, return stego as HxWxC uint8 array."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        model._encoder_svc.encode(cover_path, tmp_path, message)
        return iio.imread(tmp_path)
    finally:
        os.unlink(tmp_path)


def residual_gray(orig: np.ndarray, stego: np.ndarray, amplify: int = AMPLIFY) -> np.ndarray:
    """Per-pixel absolute difference, amplified, collapsed to grayscale."""
    diff = np.abs(orig.astype(np.int32) - stego.astype(np.int32))
    gray = diff.mean(axis=2)          # average across RGB channels
    return np.clip(gray * amplify, 0, 255).astype(np.uint8)


def to_display(arr: np.ndarray) -> Image.Image:
    img = Image.fromarray(arr)
    return img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)


# ── Load models ───────────────────────────────────────────────────────────────
print("Loading Dense model…")
dense_model    = SteganoGAN.load(DENSE_CKPT,    gpu=False, verbose=False)

print("Loading Edge-UNet model…")
unet_model     = SteganoGAN.load(EDGEUNET_CKPT, gpu=False, verbose=False)

print("Loading Edge-ASPP model…")
aspp_model     = SteganoGAN.load(EDGEASPP_CKPT, gpu=False, verbose=False)

# ── Run inference via EncoderService ─────────────────────────────────────────
print("Encoding with Dense…")
dense_arr = encode_via_service(dense_model, COVER_PATH, MESSAGE)

print("Encoding with Edge-UNet…")
unet_arr  = encode_via_service(unet_model,  COVER_PATH, MESSAGE)

print("Encoding with Edge-ASPP…")
aspp_arr  = encode_via_service(aspp_model,  COVER_PATH, MESSAGE)

# ── Load original at full resolution ─────────────────────────────────────────
orig_arr = iio.imread(COVER_PATH)

# ── Compute residuals at full resolution (before any resize) ─────────────────
dense_resid_arr = residual_gray(orig_arr, dense_arr)
unet_resid_arr  = residual_gray(orig_arr, unet_arr)
aspp_resid_arr  = residual_gray(orig_arr, aspp_arr)

# ── Resize all panels to display size ────────────────────────────────────────
orig_img        = to_display(orig_arr)
dense_img       = to_display(dense_arr)
unet_img        = to_display(unet_arr)
aspp_img        = to_display(aspp_arr)
dense_resid_img = Image.fromarray(dense_resid_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
unet_resid_img  = Image.fromarray(unet_resid_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
aspp_resid_img  = Image.fromarray(aspp_resid_arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)

# ── Compose comparison canvas ─────────────────────────────────────────────────
COLS   = [orig_img, dense_img, unet_img, aspp_img,
          dense_resid_img, unet_resid_img, aspp_resid_img]
LABELS = [
    "Original",
    "Dense",
    "Edge-UNet",
    "Edge-ASPP",
    f"Dense residual ×{AMPLIFY}",
    f"Edge-UNet residual ×{AMPLIFY}",
    f"Edge-ASPP residual ×{AMPLIFY}",
]

LABEL_H = 36
PAD     = 16
N       = len(COLS)
total_w = N * DISPLAY_SIZE + (N + 1) * PAD
total_h = LABEL_H + DISPLAY_SIZE + 2 * PAD

canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))

try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except Exception:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(canvas)

for i, (img, label) in enumerate(zip(COLS, LABELS)):
    x     = PAD + i * (DISPLAY_SIZE + PAD)
    y_img = LABEL_H + PAD

    bbox   = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((x + (DISPLAY_SIZE - text_w) // 2, PAD // 2), label,
              fill=(30, 30, 30), font=font)

    panel = img.convert("RGB") if img.mode == "L" else img
    canvas.paste(panel, (x, y_img))

canvas.save(OUT_PATH)
print(f"Saved → {OUT_PATH}")
