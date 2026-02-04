#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SteganoGAN Testing Script
Load a trained model and test encoding/decoding of secret messages in images.
"""

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import argparse
from models import SteganoGAN


def test_encode_decode(steganogan, input_image, output_image, message):
    """Test encoding and decoding a single message."""
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found!")
        return False

    print(f"\nEncoding message: '{message}'")
    steganogan.encode(input_image, output_image, message)
    print(f"Encoded image saved to: {output_image}")

    print(f"Decoding message from: {output_image}")
    decoded_message = steganogan.decode(output_image)
    print(f"Decoded message: '{decoded_message}'")

    if decoded_message == message:
        print("✓ Message decoded correctly!")
        return True
    else:
        print("✗ Warning: Decoded message differs from original")
        return False


def test_multiple_messages(steganogan, input_image, messages):
    """Test encoding and decoding multiple messages."""
    print("\n" + "="*60)
    print("Testing Multiple Messages")
    print("="*60)

    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found!")
        return

    results = []
    for i, message in enumerate(messages):
        test_output = f'test_output_{i}.png'

        print(f"\nTest {i+1}/{len(messages)}: '{message}'")
        steganogan.encode(input_image, test_output, message)
        decoded = steganogan.decode(test_output)

        success = decoded == message
        status = "✓" if success else "✗"
        print(f"{status} Decoded: '{decoded}'")
        results.append(success)

        # Clean up test file
        if os.path.exists(test_output):
            os.remove(test_output)

    # Summary
    total = len(results)
    passed = sum(results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)


def display_model_info(steganogan):
    """Display model architecture information."""
    print("\n" + "="*60)
    print("Model Information")
    print("="*60)
    print(f"  Encoder type: {type(steganogan.encoder).__name__}")
    print(f"  Decoder type: {type(steganogan.decoder).__name__}")
    print(f"  Data depth: {steganogan.data_depth}")
    print(f"  Device: {steganogan.device}")
    print(f"  GPU acceleration: {steganogan.gpu}")

    # Count parameters
    encoder_params = sum(p.numel() for p in steganogan.encoder.parameters())
    decoder_params = sum(p.numel() for p in steganogan.decoder.parameters())
    total_params = encoder_params + decoder_params

    print(f"\nModel Parameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Total: {total_params:,}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test SteganoGAN model')
    parser.add_argument('--model', type=str, default='models/weights.steg',
                        help='Path to trained model weights')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use GPU for testing if available')
    parser.add_argument('--input', type=str, default='input.png',
                        help='Input image path')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Output image path')
    parser.add_argument('--message', type=str, default='This is a super secret message!',
                        help='Secret message to encode')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    parser.add_argument('--test-multiple', action='store_true',
                        help='Run multiple message tests')

    args = parser.parse_args()

    print("="*60)
    print("SteganoGAN Testing")
    print("="*60)
    print(f"\nModel path: {args.model}")
    print(f"Input image: {args.input}")
    print(f"Output image: {args.output}")
    print(f"Secret message: {args.message}")

    # Load the model
    print(f"\nLoading model...")
    steganogan = SteganoGAN.load(
        path=args.model,
        gpu=args.gpu,
        verbose=args.verbose
    )
    print("Model loaded successfully!")

    # Display model information
    display_model_info(steganogan)

    # Test encoding and decoding
    print("\n" + "="*60)
    print("Encoding/Decoding Test")
    print("="*60)
    test_encode_decode(steganogan, args.input, args.output, args.message)

    # Test with multiple messages if requested
    if args.test_multiple:
        test_messages = [
            "Hello World!",
            "SteganoGAN is working!",
            "1234567890",
            "Testing multiple messages",
            "Short msg",
        ]
        test_multiple_messages(steganogan, args.input, test_messages)

    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
