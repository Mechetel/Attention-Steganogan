import os
from imageio import imread
from PIL import Image
import torch

class SampleGenerator:
    """Generates sample images during training."""

    def __init__(self, encoder, payload_generator, device):
        self.encoder = encoder
        self.payload_generator = payload_generator
        self.device = device

    def generate_samples(self, samples_path, epoch, text_to_encode, data_depth):
        """Generate and save sample encoded images."""
        callback_images_path = os.path.join('data', 'callback_images')
        if not os.path.exists(callback_images_path):
            os.makedirs(callback_images_path)
            raise ValueError("callback_images directory not found. Please add images to generate samples.")
        image_filenames = sorted(os.listdir(callback_images_path))
        if len(image_filenames) < 8:
            raise ValueError("Expected at least 8 generated images in callback_images")

        reshaped_tensors = []
        original_images = []

        for filename in image_filenames[:8]:
            path = os.path.join(callback_images_path, filename)

            image = imread(path, pilmode='RGB') / 127.5 - 1.0
            tensor = torch.FloatTensor(image).permute(2, 0, 1)

            if text_to_encode:
                cover = tensor.unsqueeze(0).to(self.device)
                cover_size = cover.size()

                payload = self.payload_generator.make_payload(
                    cover_size[3], cover_size[2], data_depth, text_to_encode
                )
                payload = payload.to(self.device)

                encoded_tensor = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)
                tensor = encoded_tensor.squeeze(0).to(self.device)

            original_tensor = tensor.clamp(-1.0, 1.0)
            original_tensor = ((original_tensor + 1.0) / 2.0 * 255.0).byte()
            original_image = Image.fromarray(original_tensor.permute(1, 2, 0).cpu().numpy())
            original_images.append((filename, original_image))

            resized_tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(360, 360), mode='bilinear', align_corners=False
            ).squeeze(0)
            reshaped_tensors.append(resized_tensor)

        self._create_and_save_grid(reshaped_tensors, samples_path, epoch)

    def _create_and_save_grid(self, reshaped_tensors, samples_path, epoch):
        """Create and save image grid."""
        batch = torch.stack(reshaped_tensors).clamp(-1.0, 1.0)
        batch = ((batch + 1.0) / 2.0 * 255.0).byte()
        images = [Image.fromarray(t.permute(1, 2, 0).cpu().numpy()) for t in batch]

        grid_cols = 4
        grid_rows = 2
        gap = 20
        img_w, img_h = images[0].size
        total_w = grid_cols * img_w + (grid_cols - 1) * gap
        total_h = grid_rows * img_h + (grid_rows - 1) * gap
        grid_img = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))

        for idx, img in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * (img_w + gap)
            y = row * (img_h + gap)
            grid_img.paste(img, (x, y))

        grid_filename = f'grid_epoch_{epoch}.png'
        grid_output_path = os.path.join(samples_path, grid_filename)
        grid_img.save(grid_output_path)