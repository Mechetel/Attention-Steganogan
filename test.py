#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SteganoGAN inference test script.

Loads a trained model and verifies encode → decode round-trips.
"""

import os
import sys
import argparse
from typing import List

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from steganogan import SteganoGAN


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_model_info(model: SteganoGAN) -> None:
    p = model.parameter_count
    print(f"\n{'=' * 60}")
    print("Model")
    print("=" * 60)
    print(f"  {model}")
    print(f"\n  Architecture:")
    print(f"    Encoder : {model.encoder}")
    print(f"    Decoder : {model.decoder}")
    if model.critic is not None:
        print(f"    Critic  : {model.critic}")
    print(f"\n  Parameters:")
    print(f"    Encoder : {p['encoder']:>12,}")
    print(f"    Decoder : {p['decoder']:>12,}")
    if model.critic is not None:
        print(f"    Critic  : {p['critic']:>12,}")
    print(f"    Total   : {p['total']:>12,}")


def test_roundtrip(
    model:        SteganoGAN,
    cover_path:   str,
    output_path:  str,
    message:      str,
) -> bool:
    """Encode *message* into *cover_path*, decode it, return success flag."""
    if not os.path.exists(cover_path):
        print(f"  ✗ Cover image not found: {cover_path!r}")
        return False

    model.encode(cover_path, output_path, message)
    decoded = model.decode(output_path)
    ok = decoded == message
    print(f"  {'✓' if ok else '✗'}  '{message[:60]}'"
          + ("" if ok else f"\n      got: '{decoded[:60]}'"))
    return ok


def test_batch(
    model:      SteganoGAN,
    image_dir:  str,
    messages:   List[str],
) -> None:
    """Run a round-trip test for each image in *image_dir* with a different message."""
    print(f"\n{'=' * 60}")
    print(f"Batch test  (dir: {image_dir!r})")
    print("=" * 60)

    if not os.path.exists(image_dir):
        print(f"  ✗ Image directory not found: {image_dir!r}")
        return

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not images:
        print(f"  ✗ No images found in {image_dir!r}")
        return

    print(f"  Found {len(images)} image(s), {len(messages)} message(s)")

    results = []
    for i, cover_path in enumerate(images):
        msg = messages[i % len(messages)]
        tmp = f"_test_tmp_{i}.png"
        fname = os.path.basename(cover_path)
        print(f"\n  [{i+1}/{len(images)}] {fname}")
        print(f"    msg : '{msg[:60]}'")
        try:
            ok = test_roundtrip(model, cover_path, tmp, msg)
            results.append(ok)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    passed = sum(results)
    total  = len(results)
    print(f"\n  Result: {passed}/{total} passed")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SteganoGAN inference test")
    p.add_argument("--model",         default="models/weights.steg")
    p.add_argument("--gpu",           action="store_true", default=False)
    p.add_argument("--depth",         type=int, default=None, help="Override data_depth (DepthAgnosticEncoder/Decoder only)")
    p.add_argument("--input",         default="input.png")
    p.add_argument("--output",        default="output.png")
    p.add_argument("--message",       default="This is a super secret message!")
    p.add_argument("--verbose",       action="store_true", default=True)
    p.add_argument("--test-multiple", action="store_true")
    p.add_argument("--callback-dir",  default="data/callback_images")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("SteganoGAN Inference Test")
    print("=" * 60)
    print(f"  model  : {args.model}")
    print(f"  input  : {args.input}")
    print(f"  output : {args.output}")

    model = SteganoGAN.load(args.model, gpu=args.gpu, verbose=args.verbose)
    if args.depth is not None:
        model.set_depth(args.depth)
        print(f"  depth  : {args.depth}  (overridden)")
    print_model_info(model)

    print(f"\n{'=' * 60}")
    print("Round-trip test")
    print("=" * 60)
    test_roundtrip(model, args.input, args.output, args.message)

    if args.test_multiple:
        test_batch(model, args.callback_dir, messages=[
            "Hello World!",
            "SteganoGAN is working!",
            "1234567890",
            "Testing multiple messages",
            "Short msg",
            "Another secret payload",
            "Deep learning steganography",
            "PhD dissertation test",
        ])

    print(f"\n{'=' * 60}")
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
