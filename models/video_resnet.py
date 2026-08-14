import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class VideoResNetEncoder(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = r3d_18(weights=weights)
        
        # Remove the fully connected classification head
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        
        # Output dim of R3D_18 before FC is 512.
        resnet_out_dim = 512
        
        if resnet_out_dim != output_dim:
            self.proj = nn.Linear(resnet_out_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # x: (B, C, T, H, W)
        features = self.feature_extractor(x) # (B, 512, 1, 1, 1) usually if pooled
        features = features.flatten(1) # (B, 512)
        # Sequence format for MMFusion
        features = features.unsqueeze(1) # (B, 1, 512)
        
        features = self.proj(features)
        return features
