import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
from models.ltae import LTAE2d

class VideoAlexNetEncoder(nn.Module):
    def __init__(self, pretrained=True, mlp=[256, 256]):
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        
        # Feature extractor from AlexNet (returns 256 channels)
        self.feature_extractor = alexnet(weights=weights).features
        
        # LTAE for temporal attention
        self.temporal_encoder = LTAE2d(
            in_channels=256,
            d_model=256,
            mlp=mlp,
            positional_encoding="normal"
        )

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        # 1. Fold time into batch: (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # 2. Extract spatial features
        features = self.feature_extractor(x) # (B*T, 256, H_out, W_out)
        _, D, H_out, W_out = features.shape
        
        # 3. Unfold time dimension: (B, T, D, H_out, W_out)
        features = features.reshape(B, T, D, H_out, W_out)
        
        # 4. Permute to match LTAE input format (B, D, T, H_out, W_out)
        features = features.permute(0, 2, 1, 3, 4)
        
        # 5. Temporal encoding
        time_linear = torch.arange(T, device=features.device).unsqueeze(0).repeat(B, 1)
        batch_positions = {"time_linear": time_linear}
        
        # Aggregate over time
        features = self.temporal_encoder(features, batch_positions=batch_positions) # (B, mlp[-1], H_out, W_out)
        
        return features
