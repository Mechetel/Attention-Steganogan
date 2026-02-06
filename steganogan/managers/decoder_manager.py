from collections import Counter
import os
import torch
from skimage.io import imread
from ..utils import bits_to_bytearray, bytearray_to_text

class DecoderManager:
    """Manages image decoding operations."""

    def __init__(self, decoder, device, verbose=False):
        self.decoder = decoder
        self.device = device
        self.verbose = verbose

    def decode(self, image_path):
        """Decode hidden message from image."""
        if not os.path.exists(image_path):
            raise ValueError('Unable to read %s.' % image_path)

        image = imread(image_path, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0

        candidates = Counter()
        bits = image.data.int().cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        if len(candidates) == 0:
            raise ValueError('Failed to find message.')

        candidate, count = candidates.most_common(1)[0]

        if self.verbose:
            print(f'Decoding completed. Message found {count} times.')

        return candidate