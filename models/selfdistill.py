"""Heads and predictors for the multimodal self-distillation family.

Both objectives share one skeleton (an EMA teacher on the full multimodal view,
students on the M+1 modality masks) and differ only in what sits between the student
representation and the target:

    byol  a two-layer MLP predictor, cosine loss on pooled embeddings
    dino  a shared prototype head, cross-entropy on centered/sharpened teacher logits
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class BYOLPredictor(nn.Module):
    """BYOL's asymmetry lives here: only the student passes through it."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOHead(nn.Module):
    """Projection to `out_dim` prototypes, with DINO's normalised last layer.

    `out_dim` is the one hyper-parameter that does not transfer from the image
    setting: DINO uses 65536 prototypes on ImageNet, which is meaningless for the
    40-d embeddings of the MultiBench affect datasets. It is left in the config.
    """

    def __init__(self, in_dim: int, out_dim: int = 2048, hidden_dim: int = 1024,
                 bottleneck_dim: int = 128, n_layers: int = 3):
        super().__init__()
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)
        # DINO's weight-normalised last layer with the gain frozen at 1 is exactly a
        # row-normalised weight matrix. Written directly rather than through
        # nn.utils.weight_norm, whose computed `weight` is not a graph leaf and so
        # makes the module impossible to deepcopy into an EMA teacher.
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return F.linear(x, F.normalize(self.last_layer.weight, dim=1, p=2))
