# -*- coding: utf-8 -*-

import zlib
from math import exp
from typing import List, Optional, Union

import torch
from reedsolo import RSCodec
from torch.nn.functional import conv2d


class MessageEncoder:
    """Handles encoding and decoding of text messages with compression and error correction."""

    def __init__(self, rs_symbols: int = 250) -> None:
        self.rs: RSCodec = RSCodec(rs_symbols)

    def text_to_bits(self, text: str) -> List[int]:
        """Convert text to a list of ints in {0, 1}."""
        return self.bytearray_to_bits(self.text_to_bytearray(text))

    def bits_to_text(self, bits: List[int]) -> Union[str, bool]:
        """Convert a list of ints in {0, 1} to text."""
        return self.bytearray_to_text(self.bits_to_bytearray(bits))

    def bytearray_to_bits(self, x: Union[bytes, bytearray]) -> List[int]:
        """Convert bytearray to a list of bits."""
        result: List[int] = []
        for i in x:
            bits = bin(i)[2:]
            bits = '00000000'[len(bits):] + bits
            result.extend([int(b) for b in bits])
        return result

    def bits_to_bytearray(self, bits: List[int]) -> bytearray:
        """Convert a list of bits to a bytearray."""
        ints = []
        for b in range(len(bits) // 8):
            byte = bits[b * 8:(b + 1) * 8]
            ints.append(int(''.join([str(bit) for bit in byte]), 2))
        return bytearray(ints)

    def text_to_bytearray(self, text: str) -> bytearray:
        """Compress text and add Reed-Solomon error correction."""
        assert isinstance(text, str), "expected a string"
        x = zlib.compress(text.encode("utf-8"))
        x = self.rs.encode(bytearray(x))
        return x

    def bytearray_to_text(self, x: Union[bytes, bytearray]) -> Union[str, bool]:
        """Apply error correction and decompress bytes to text."""
        try:
            assert isinstance(x, (bytes, bytearray)), "expected bytes or bytearray"
            decoded = self.rs.decode(x)
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            return zlib.decompress(bytes(decoded)).decode("utf-8")
        except BaseException:
            return False


class SSIMCalculator:
    """Calculates Structural Similarity Index (SSIM) between images."""

    @staticmethod
    def gaussian(window_size: int, sigma: float) -> torch.Tensor:
        """Compute a 1-D Gaussian window of length `window_size` and std `sigma`.

        https://en.wikipedia.org/wiki/Window_function#Gaussian_window
        """
        _exp = [
            exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
        gauss = torch.Tensor(_exp)
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size: int, channel: int) -> torch.Tensor:
        """Create a 2-D Gaussian window expanded across `channel` channels."""
        _1D_window = SSIMCalculator.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    @staticmethod
    def _ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        size_average: bool = True,
    ) -> torch.Tensor:
        """Compute raw SSIM map and optionally average it."""
        padding_size = window_size // 2

        mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
        mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

        mu1_sq  = mu1.pow(2)
        mu2_sq  = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
        sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
        sigma12   = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = (
            ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
            / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        )

        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

    @staticmethod
    def calculate(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        size_average: bool = True,
    ) -> torch.Tensor:
        """Compute SSIM between two image tensors."""
        (_, channel, _, _) = img1.size()
        window = SSIMCalculator.create_window(window_size, channel)
        window = window.to(img1.device).type_as(img1)
        return SSIMCalculator._ssim(img1, img2, window, window_size, channel, size_average)


# ── Backward-compatible module-level aliases ──────────────────────────────────

_default_encoder = MessageEncoder()


def text_to_bits(text: str) -> List[int]:
    return _default_encoder.text_to_bits(text)


def bits_to_text(bits: List[int]) -> Union[str, bool]:
    return _default_encoder.bits_to_text(bits)


def bytearray_to_bits(x: Union[bytes, bytearray]) -> List[int]:
    return _default_encoder.bytearray_to_bits(x)


def bits_to_bytearray(bits: List[int]) -> bytearray:
    return _default_encoder.bits_to_bytearray(bits)


def text_to_bytearray(text: str) -> bytearray:
    return _default_encoder.text_to_bytearray(text)


def bytearray_to_text(x: Union[bytes, bytearray]) -> Union[str, bool]:
    return _default_encoder.bytearray_to_text(x)


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    return SSIMCalculator.calculate(img1, img2, window_size, size_average)


def first_element(storage: object, loc: object) -> object:
    """Returns the first element of two (pickle map_location helper)."""
    return storage
