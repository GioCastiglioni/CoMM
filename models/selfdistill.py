"""Heads, predictors and the token-mask sampler for the multimodal self-distillation family.

The three objectives share one skeleton (an EMA teacher on the full multimodal view,
students on the M+1 modality masks) and differ only in what sits between the student
representation and the target:

    byol  a two-layer MLP predictor, cosine loss on pooled embeddings
    dino  a shared prototype head, cross-entropy on centered/sharpened teacher logits
    jepa  a transformer predictor conditioned on (modality, position), smooth-L1 on tokens
"""
import math
from typing import List, Optional, Sequence, Tuple

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


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class JEPAPredictor(nn.Module):
    """Predict target-token representations from context tokens.

    Queries carry the identity of what is being asked for -- which modality and
    which position -- because without it the predictor cannot tell the targets
    apart and its best output is their average, which is the collapse direction.
    """

    def __init__(self, dim: int, n_modalities: int, max_len: int = 512,
                 depth: int = 2, n_heads: int = 8, per_modality: bool = True):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} is not divisible by n_heads={n_heads}")
        self.per_modality = per_modality
        n_pred = n_modalities if per_modality else 1
        self.blocks = nn.ModuleList([
            nn.ModuleList([_Block(dim, n_heads) for _ in range(depth)])
            for _ in range(n_pred)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_pred)])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mod_embed = nn.Parameter(torch.zeros(n_modalities, dim))
        self.pos_embed = nn.Parameter(torch.zeros(max_len, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.mod_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _tag(self, x: torch.Tensor, mod: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return x + self.mod_embed[mod].unsqueeze(0) + self.pos_embed[pos].unsqueeze(0)

    def forward(self, ctx: torch.Tensor, ctx_mod: torch.Tensor, ctx_pos: torch.Tensor,
                tgt_mod: torch.Tensor, tgt_pos: torch.Tensor,
                branch: int = 0) -> torch.Tensor:
        """
        Args:
            ctx: (B, n_ctx, D) student tokens.
            ctx_mod / ctx_pos: (n_ctx,) modality id and within-modality position.
            tgt_mod / tgt_pos: (n_tgt,) the same for the tokens to predict.
            branch: which predictor to use when `per_modality` is set.
        Returns: (B, n_tgt, D)
        """
        b, n_tgt = ctx.shape[0], tgt_mod.shape[0]
        idx = branch if self.per_modality else 0
        x = self._tag(ctx, ctx_mod, ctx_pos)
        q = self._tag(self.mask_token.expand(b, n_tgt, -1), tgt_mod, tgt_pos)
        x = torch.cat([x, q], dim=1)
        for blk in self.blocks[idx]:
            x = blk(x)
        return self.norm[idx](x[:, -n_tgt:])


def _uniform(lo: float, hi: float, generator: Optional[torch.Generator]) -> float:
    return float(torch.empty(1).uniform_(lo, hi, generator=generator).item())


def _block(grid: Tuple[int, int], scale: float, aspect: float,
           generator: Optional[torch.Generator]) -> torch.Tensor:
    """One rectangle of the requested area fraction, as a flat boolean mask."""
    h, w = grid
    area = max(1.0, scale * h * w)
    bh = max(1, min(h, int(round(math.sqrt(area / aspect)))))
    bw = max(1, min(w, int(round(math.sqrt(area * aspect)))))
    top = int(torch.randint(0, h - bh + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - bw + 1, (1,), generator=generator).item())
    m = torch.zeros(h, w, dtype=torch.bool)
    m[top:top + bh, left:left + bw] = True
    return m.flatten()


def _span(length: int, scale: float,
          generator: Optional[torch.Generator]) -> torch.Tensor:
    """The 1-D analogue: one contiguous span.

    Contiguous rather than scattered, because isolated masked positions are
    recoverable by interpolating their neighbours.
    """
    n = max(1, min(length, int(round(scale * length))))
    start = int(torch.randint(0, length - n + 1, (1,), generator=generator).item())
    m = torch.zeros(length, dtype=torch.bool)
    m[start:start + n] = True
    return m


def sample_ijepa_masks(seq_lens: Sequence[int],
                       mode: str = "block",
                       n_targets: int = 4,
                       target_scale: Tuple[float, float] = (0.15, 0.2),
                       target_aspect: Tuple[float, float] = (0.75, 1.5),
                       context_scale: Tuple[float, float] = (0.85, 1.0),
                       grids: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
                       generator: Optional[torch.Generator] = None,
                       ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
    """I-JEPA's sampling scheme, per modality: `n_targets` blocks plus one context.

    Faithful to the original in the parts that matter:
      * `n_targets` target blocks at scale `target_scale` and aspect `target_aspect`;
        they may overlap *each other*, and each is predicted on its own;
      * one context block at scale `context_scale`, from which every target patch is
        then **removed** -- so context and targets are disjoint by construction, not
        by luck. The function asserts it before returning.

    One split per modality, shared across the batch (so token tensors stay
    rectangular) and shared across the M+1 student views (so modality availability
    is the only thing separating them -- a different mask per view would hand the
    multimodal branch an easier task for reasons unrelated to having both
    modalities).

    Returns one `(context_idx, [target_idx, ...])` pair per modality.
    """
    out = []
    for i, length in enumerate(seq_lens):
        grid = grids[i] if grids is not None and i < len(grids) else None
        if mode == "block" and grid is None:
            side = int(round(math.sqrt(length)))
            grid = (side, side) if side * side == length else None
        use_block = mode == "block" and grid is not None

        targets, union = [], torch.zeros(length, dtype=torch.bool)
        for _ in range(n_targets):
            s = _uniform(*target_scale, generator=generator)
            m = (_block(grid, s, _uniform(*target_aspect, generator=generator), generator)
                 if use_block else _span(length, s, generator))
            targets.append(m)
            union |= m

        s_ctx = _uniform(*context_scale, generator=generator)
        ctx = (_block(grid, s_ctx, 1.0, generator) if use_block
               else _span(length, s_ctx, generator))
        # I-JEPA's overlap removal: this is what makes the two sets disjoint
        ctx = ctx & ~union

        # neither side may be empty: nothing to predict from, or nothing to predict
        if not ctx.any():
            free = torch.nonzero(~union).flatten()
            if free.numel() == 0:                    # every patch is a target
                drop = torch.nonzero(targets[-1]).flatten()[0]
                for m in targets:
                    m[drop] = False
                union = torch.zeros(length, dtype=torch.bool)
                for m in targets:
                    union |= m
                free = torch.nonzero(~union).flatten()
            ctx[free[0]] = True
        targets = [m for m in targets if m.any()]
        if not targets:
            spare = torch.nonzero(ctx).flatten()[-1]
            ctx[spare] = False
            m = torch.zeros(length, dtype=torch.bool)
            m[spare] = True
            targets = [m]

        ctx_idx = torch.nonzero(ctx).flatten()
        tgt_idx = [torch.nonzero(m).flatten() for m in targets]
        for t in tgt_idx:
            assert not bool((ctx[t]).any()), "context and target overlap"
        out.append((ctx_idx, tgt_idx))
    return out
