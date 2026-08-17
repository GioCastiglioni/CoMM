import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class AlexNetMultispectralEncoder(nn.Module):
    """AlexNet backbone adapted for multispectral imagery."""
    def __init__(self, in_channels: int, latent_dim: int = 512, dropout: float = 0.5, global_pool: str = ""):
        super().__init__()
        assert global_pool in {"avg", ""}
        
        # Load standard pretrained AlexNet
        base_model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = base_model.features
        
        # Modify the first convolutional layer if in_channels is not 3
        if in_channels != 3:
            old_conv = self.features[0]
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels, 
                                 kernel_size=old_conv.kernel_size, 
                                 stride=old_conv.stride, 
                                 padding=old_conv.padding, 
                                 bias=old_conv.bias is not None)
            
            # Initialize with random weights (Kaiming Normal is typical for ReLU)
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)
                
            # If in_channels == 13 (Sentinel-2), map RGB to B2, B3, B4 (indices 1, 2, 3)
            # Sentinel-2 bands: B1, B2(B), B3(G), B4(R), ...
            # Wait, the user said: "las bandas B2, B3 y B4 son G R y B respectivamente"
            # PyTorch models pretrained on ImageNet expect RGB order.
            # So ImageNet R -> B4 (idx 3), ImageNet G -> B3 (idx 2), ImageNet B -> B2 (idx 1).
            if in_channels == 13:
                with torch.no_grad():
                    # RGB to B4, B3, B2
                    new_conv.weight[:, 3, :, :] = old_conv.weight[:, 0, :, :] # R -> B4
                    new_conv.weight[:, 2, :, :] = old_conv.weight[:, 1, :, :] # G -> B3
                    new_conv.weight[:, 1, :, :] = old_conv.weight[:, 2, :, :] # B -> B2
            
            # If in_channels == 2 (Sentinel-1 VV, VH), we can just average RGB weights or keep random
            # Random is fine as per usual standard for SAR, but let's average just in case
            if in_channels == 2:
                with torch.no_grad():
                    avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 0:1, :, :] = avg_weight
                    new_conv.weight[:, 1:2, :, :] = avg_weight

            self.features[0] = new_conv

        self.classifier = nn.Linear(256 * 6 * 6, latent_dim)
        self.global_pool = global_pool
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.features(x)
        if self.global_pool == "avg":
            x = self.dropout(x)
            x = x.flatten(1)
            x = self.classifier(x)
            x = x.unsqueeze(1)
        return x
