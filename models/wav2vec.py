import torch
import torch.nn as nn
import torchaudio

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.encoder = bundle.get_model()

    def forward(self, x):
        # x is assumed to be (batch_size, channels, time)
        # Wav2Vec2 expects (batch_size, time) for single channel
        if x.dim() == 3:
            # average over channels if more than 1, or just squeeze
            x = x.mean(dim=1)
            
        features, _ = self.encoder(x)  # (batch_size, frames, wav2vec_out_dim=768)
        
        return features

