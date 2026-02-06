import torch
from ..utils import text_to_bits

class PayloadGenerator:
    """Handles payload generation for encoding."""

    def __init__(self, device):
        self.device = device

    def random_data(self, cover, data_depth):
        """Generate random binary payload matching cover dimensions."""
        N, _, H, W = cover.size()
        return torch.zeros((N, data_depth, H, W), device=self.device).random_(0, 2)

    def make_payload(self, width, height, depth, text):
        """Create payload from text message."""
        message = text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width).to(self.device)
