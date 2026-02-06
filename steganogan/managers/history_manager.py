import json
import os

class HistoryManager:
    """Manages training history and metrics logging."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.history = []
        self.metrics_path = os.path.join(log_dir, 'metrics.log') if log_dir else None
        self._load_existing_history()

    def _load_existing_history(self):
        """Load existing metrics from log file."""
        if self.metrics_path and os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as metrics_file:
                    self.history = json.load(metrics_file)
                    print(f"Loaded {len(self.history)} previous epochs")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load existing metrics.log: {e}")
                self.history = []

    def append_and_save(self, metrics):
        """Append metrics to history and save to file."""
        self.history.append(metrics)
        if self.metrics_path:
            with open(self.metrics_path, 'w') as metrics_file:
                json.dump(self.history, metrics_file, indent=4)

    def print_metrics(self, metrics):
        """Print formatted metrics."""
        print(f"\nEpoch {metrics['epoch']} Metrics:")
        print(f"  Train - Enc MSE: {metrics['train.encoder_mse']:.6f}, "
              f"Dec Loss: {metrics['train.decoder_loss']:.4f}, "
              f"Dec Acc: {metrics['train.decoder_acc']:.4f}")
        print(f"  Val   - Enc MSE: {metrics['val.encoder_mse']:.6f}, "
              f"Dec Loss: {metrics['val.decoder_loss']:.4f}, "
              f"Dec Acc: {metrics['val.decoder_acc']:.4f}")
        print(f"  Val   - SSIM: {metrics['val.ssim']:.4f}, "
              f"PSNR: {metrics['val.psnr']:.2f}, "
              f"RSBPP: {metrics['val.rsbpp']:.4f}")
