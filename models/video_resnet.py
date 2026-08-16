import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class VideoResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = r3d_18(weights=weights)
        
        # Remove the fully connected classification head
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        # x: (B, C, T, H, W)
        features = self.feature_extractor(x) # (B, 512, 1, 1, 1) usually if pooled
        features = features.flatten(1) # (B, 512)

        return features.unsqueeze(1)    # (B, 1, 512)
