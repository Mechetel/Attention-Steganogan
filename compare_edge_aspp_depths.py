"""
Loads Edge-ASPP SteganoGAN checkpoints for D=1..4, runs inference,
and produces a comparison grid:
  Row 1: Original | Stego D=1 | Stego D=2 | Stego D=3 | Stego D=4
  Row 2: (empty)  | Residual ×50 D=1 | ... | Residual ×50 D=4
"""
import sys, os, tempfile, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda.amp")

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from steganogan.engine import SteganoGAN

BASE = os.path.dirname(__file__)

CKPTS = {
    1: os.path.join(BASE, "models/div2k/edge_aspp_edge_aware_dense/1774269623-d1/32.rsbpp-0.971016.p"),
    2: os.path.join(BASE, "models/div2k/edge_aspp_edge_aware_dense/1774253954-d2/32.rsbpp-1.891735.p"),
    3: os.path.join(BASE, "models/div2k/edge_aspp_edge_aware_dense/1774256915-d3/32.rsbpp-2.405477.p"),
    4: os.path.join(BASE, "models/div2k/edge_aspp_edge_aware_dense/1774272817-d4/32.rsbpp-1.787670.p"),
}

COVER_PATH   = os.path.join(BASE, "data/callback_images/0314.png")
OUT_PATH     = os.path.join(BASE, "comparison_edge_aspp_depths_50x.png")
MESSAGE      = "Hello, SteganoGAN!"
DISPLAY_SIZE = 360
AMPLIFY      = 50


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
    gray = diff.mean(axis=2)
    return np.clip(gray * amplify, 0, 255).astype(np.uint8)


def to_display(arr: np.ndarray) -> Image.Image:
    img = Image.fromarray(arr)
    return img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)


# ── Load models & encode ─────────────────────────────────────────────────────
orig_arr = iio.imread(COVER_PATH)
stego_arrs = {}
for d, ckpt in CKPTS.items():
    print(f"Loading Edge-ASPP D={d}…")
    model = SteganoGAN.load(ckpt, gpu=False, verbose=False)
    print(f"Encoding with D={d}…")
    stego_arrs[d] = encode_via_service(model, COVER_PATH, MESSAGE)
    del model

# ── Compute residuals at full resolution ─────────────────────────────────────
resid_arrs = {d: residual_gray(orig_arr, stego_arrs[d]) for d in CKPTS}

# ── Build grid ───────────────────────────────────────────────────────────────
NCOLS = 5  # original + 4 depths
NROWS = 2

LABEL_H = 36
PAD     = 16
total_w = NCOLS * DISPLAY_SIZE + (NCOLS + 1) * PAD
total_h = NROWS * (LABEL_H + DISPLAY_SIZE + PAD) + PAD

canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))

try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except Exception:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(canvas)

# Row 1: Original + stego images
row1_imgs   = [to_display(orig_arr)] + [to_display(stego_arrs[d]) for d in range(1, 5)]
row1_labels = ["Original"] + [f"Edge-ASPP D={d}" for d in range(1, 5)]

# Row 2: empty placeholder + residuals
row2_imgs   = [None] + [
    Image.fromarray(resid_arrs[d]).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
    for d in range(1, 5)
]
row2_labels = [""] + [f"Residual ×{AMPLIFY} D={d}" for d in range(1, 5)]

for row_idx, (imgs, labels) in enumerate([(row1_imgs, row1_labels), (row2_imgs, row2_labels)]):
    for col_idx, (img, label) in enumerate(zip(imgs, labels)):
        x = PAD + col_idx * (DISPLAY_SIZE + PAD)
        y_label = PAD + row_idx * (LABEL_H + DISPLAY_SIZE + PAD)
        y_img   = y_label + LABEL_H

        if label:
            bbox   = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (DISPLAY_SIZE - text_w) // 2, y_label + PAD // 4),
                      label, fill=(30, 30, 30), font=font)

        if img is not None:
            panel = img.convert("RGB") if img.mode == "L" else img
            canvas.paste(panel, (x, y_img))

canvas.save(OUT_PATH)
print(f"Saved → {OUT_PATH}")
