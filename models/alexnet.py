from torchvision.models import AlexNet
import torch.nn as nn
import torch

class AlexNetEncoder(AlexNet):
    """AlexNet backbone for features representation learning."""
    def __init__(self, latent_dim: int = 512, dropout: float = 0.5, global_pool: str = "avg", in_channels: int = 3):
        assert global_pool in {"avg", ""}
        super().__init__(dropout=dropout)
        
        if in_channels != 3:
            # Replace the first convolution layer to accept different number of channels
            self.features[0] = nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2)
            
        self.classifier = nn.Linear(256 * 6 * 6, latent_dim)
        self.global_pool = global_pool
        if self.global_pool == "":
            # `forward` returns the feature map without reaching the classifier, so it
            # gets no gradient and DDP errors on it. Frozen rather than deleted: deleting
            # it changes the state_dict and drops the RNG draw this Linear takes at
            # construction, which would shift every initialisation made afterwards.
            self.classifier.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool == "avg":
            return super().forward(x)
        return self.features(x)