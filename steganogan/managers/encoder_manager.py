import torch
from imageio import imread, imwrite

class EncoderManager:
    """Manages image encoding operations."""

    def __init__(self, encoder, payload_generator, data_depth, device, verbose=False):
        self.encoder = encoder
        self.payload_generator = payload_generator
        self.data_depth = data_depth
        self.device = device
        self.verbose = verbose

    def encode(self, cover_path, output_path, text):
        """Encode text message into cover image."""
        cover = imread(cover_path, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover_size = cover.size()
        payload = self.payload_generator.make_payload(
            cover_size[3], cover_size[2], self.data_depth, text
        )

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output_path, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')