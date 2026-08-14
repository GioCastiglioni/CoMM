import torch
import torch.nn as nn
import torchaudio

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.encoder = bundle.get_model()
        
        # Determine the output dimension of the Wav2Vec2 model
        # For WAV2VEC2_BASE, it's 768.
        wav2vec_out_dim = 768
        
        # A linear projection to match the desired embedding dimension
        self.proj = nn.Linear(wav2vec_out_dim, output_dim)

    def forward(self, x):
        # x is assumed to be (batch_size, channels, time)
        # Wav2Vec2 expects (batch_size, time) for single channel
        if x.dim() == 3:
            # average over channels if more than 1, or just squeeze
            x = x.mean(dim=1)
            
        features, _ = self.encoder(x)
        # features is (batch_size, frames, wav2vec_out_dim)
        
        # We can either mean pool over time or keep the sequence.
        # Since MMFusion uses sequence of tokens or CLS, we will pass the sequence along.
        features = self.proj(features)
        return features

