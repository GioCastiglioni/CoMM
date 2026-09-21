"""A small ViT over *native* small multispectral patches.

Written for So2Sat, whose patches are 32x32. The AlexNet encoders it replaces there
needed the input upsampled to 224 to produce more than one token, which is 49x the
pixels for no added information and was the pipeline's CPU bottleneck. Their usual
justification -- ImageNet transfer -- barely applies to this data anyway: the
multispectral adapter only remaps RGB weights for a 13-band input, so Sentinel-2's
10 bands and Sentinel-1's 2 both start their first convolution from scratch.

Tokens are patches, so unlike a convolutional readout their receptive fields are
exact and disjoint at the patch-embedding layer. At 32x32 with `patch_size=4` the
grid is 8x8 = 64 tokens, each covering 4x4 native pixels (40x40 m at Sentinel-2's
10 m resolution), against the 6x6 = 36 tokens the AlexNet readout produced.

No class token: `MMFusion` prepends its own and pools there, so this returns the
patch sequence only.
"""
from typing import Optional

import torch
import torch.nn as nn


class _Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTMultispectralEncoder(nn.Module):
    """ViT for multispectral imagery at its native resolution.

    :param in_channels: spectral bands (Sentinel-1: 2, Sentinel-2: 10)
    :param img_size: side of the input patch, in pixels
    :param patch_size: side of one token, in pixels; must divide `img_size`
    :param embed_dim: token width; match the fusion transformer's
    :param depth: transformer blocks
    :param n_heads: attention heads; `embed_dim` must be divisible by it

    Defaults give 2.7M parameters, matching the 2.47M of the `AlexNet.features`
    stack they replace, so the encoder capacity of the two arms is comparable.
    """

    def __init__(self, in_channels: int, img_size: int = 32, patch_size: int = 4,
                 embed_dim: int = 192, depth: int = 6, n_heads: int = 3,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 global_pool: str = ""):
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"patch_size={patch_size} does not divide img_size={img_size}")
        if embed_dim % n_heads:
            raise ValueError(f"embed_dim={embed_dim} is not divisible by n_heads={n_heads}")
        assert global_pool in {"avg", ""}

        self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.grid = img_size // patch_size
        n_tokens = self.grid ** 2

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # B, N, D
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.global_pool == "avg":
            return x.mean(dim=1, keepdim=True)
        return x
