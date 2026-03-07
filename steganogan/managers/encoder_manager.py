import torch
from imageio import imread, imwrite
from typing import Union


class EncoderManager:
    """Manages image encoding operations."""

    def __init__(self, encoder: torch.nn.Module, payload_generator: object,
                 data_depth: int, device: torch.device,
                 verbose: bool = False) -> None:
        self.encoder:           torch.nn.Module = encoder
        self.payload_generator: object          = payload_generator
        self.data_depth:        int             = data_depth
        self.device:            torch.device    = device
        self.verbose:           bool            = verbose

    def encode(self, cover_path: str, output_path: str, text: str) -> None:
        """
        Encode a text message into a cover image and write the stego image to disk.

        Parameters
        ----------
        cover_path  : path to the source cover image (RGB)
        output_path : path where the stego image will be saved
        text        : secret text message to embed
        """
        cover = imread(cover_path, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover_size = cover.size()
        payload = self.payload_generator.make_payload(
            cover_size[3], cover_size[2], self.data_depth, text
        )

        cover   = cover.to(self.device)
        payload = payload.to(self.device)

        raw_out = self.encoder(cover, payload)

        # Handle iterative encoder (returns list of stego images); take the final one
        if isinstance(raw_out, (list, tuple)):
            generated = raw_out[-1]
        else:
            generated = raw_out

        generated = generated[0].clamp(-1.0, 1.0)
        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output_path, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')
