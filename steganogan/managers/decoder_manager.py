from collections import Counter
import os
from typing import Optional
import torch
from skimage.io import imread
from ..utils import bits_to_bytearray, bytearray_to_text


class DecoderManager:
    """Manages image decoding operations."""

    def __init__(self, decoder: torch.nn.Module, device: torch.device,
                 verbose: bool = False) -> None:
        self.decoder: torch.nn.Module = decoder
        self.device: torch.device     = device
        self.verbose: bool            = verbose

    def decode(self, image_path: str) -> str:
        """
        Decode the hidden message from a stego image file.

        Parameters
        ----------
        image_path : path to the stego image

        Returns
        -------
        The decoded text message.

        Raises
        ------
        ValueError : if the file does not exist or no message is found.
        """
        if not os.path.exists(image_path):
            raise ValueError(f'Unable to read {image_path!r}.')

        image = imread(image_path, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0).to(self.device)

        bits  = (self.decoder(image).view(-1) > 0).data.int().cpu().numpy().tolist()

        candidates: Counter = Counter()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            text = bytearray_to_text(bytearray(candidate))
            if text:
                candidates[text] += 1

        if not candidates:
            raise ValueError('Failed to find message.')

        candidate, count = candidates.most_common(1)[0]

        if self.verbose:
            print(f'Decoding completed. Message found {count} times.')

        return candidate
