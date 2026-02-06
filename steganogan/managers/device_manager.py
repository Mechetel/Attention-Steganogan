import torch

class DeviceManager:
    """Manages device selection and model placement."""

    def __init__(self, gpu=True, verbose=False):
        self.verbose = verbose
        self.gpu_requested = gpu
        self.gpu = False
        self.device = None
        self.set_device()

    def set_device(self):
        if self.gpu_requested:
            if torch.cuda.is_available():
                self.gpu = True
                self.device = torch.device('cuda')
                if self.verbose:
                    print("Using NVIDIA GPU (CUDA).")
            elif torch.backends.mps.is_available():
                self.gpu = True
                self.device = torch.device('mps')
                if self.verbose:
                    print("Using Apple GPU (MPS).")
            else:
                self.gpu = False
                self.device = torch.device('cpu')
                if self.verbose:
                    print("GPU requested but not available. Falling back to CPU.")
        else:
            self.gpu = False
            self.device = torch.device('cpu')

            if self.verbose:
                print("Using CPU.")

    def to_device(self, *models):
        """Move models to the current device."""
        for model in models:
            model.to(self.device)

