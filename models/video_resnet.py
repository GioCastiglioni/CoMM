import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class VideoResNetEncoder(nn.Module):
    def __init__(self, pretrained=True, return_nodes=None):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = r3d_18(weights=weights)
        
        if return_nodes is None:
            return_nodes = {'layer4': 'features'}
            
        self.feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    def forward(self, x):
        features = self.feature_extractor(x)['features'] # (B, 512, 2, 7, 7)

        B, D, T, H, W = features.shape
        
        features = features.reshape(B, D*T, H, W) # (B, 1024, 7, 7)

        return features
