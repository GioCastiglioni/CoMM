"""Multimodal self-distillation: MM-BYOL, MM-DINO and MM-JEPA on one skeleton.

Read this as WoMM with two changes and nothing else:

  1. the joint (prototype) side of every pair comes from an EMA teacher instead of
     the online network, so the target is stale and detached by construction;
  2. there is no explicit regulariser -- no negatives, no variance term. Collapse
     is meant to be prevented by the teacher/student asymmetry alone.

The pairs themselves are untouched: mask `m` of view 1 against the joint of view 2,
mask `m` of view 2 against the joint of view 1, and joint against joint. That keeps
the probes, the mask structure and the biased/unbiased arms directly comparable to
the contrastive and variance families.
"""
import copy
import math
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from losses.selfdistill_loss import MMSelfDistillLoss
from models.mmfusion import MMFusion
from models.selfdistill import (BYOLPredictor, DINOHead, JEPAPredictor,
                                sample_ijepa_masks)
from pl_modules.base import BaseModel


class MMSD(BaseModel):
    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict,
                 ema_base: float = 0.996,
                 ema_final: float = 1.0,
                 predictor_kwargs: Optional[Dict] = None,
                 mask_kwargs: Optional[Dict] = None,
                 cross_view: Optional[bool] = None):
        """
        Args:
            encoder: multimodal fusion encoder (the online / student network).
            projection: MLP projector to the latent space.
            optim_kwargs: optimisation hyper-parameters.
            loss_kwargs: given to `MMSelfDistillLoss`; `objective` selects the method.
            ema_base / ema_final: teacher momentum, cosine-ramped between them.
            predictor_kwargs: for the BYOL / JEPA predictor.
            mask_kwargs: I-JEPA mask sampling (`mode`, `n_targets`, `target_scale`,
                `target_aspect`, `context_scale`).
            cross_view: whether the student of view v is matched against the teacher
                on view 1-v. Defaults per objective to what the original method does:
                True for BYOL and DINO, which are two-view methods, and False for
                I-JEPA, whose target encoder sees the *same* image as the context
                encoder and derives all of its asymmetry from masking.
        """
        super(MMSD, self).__init__(optim_kwargs)

        self.encoder = encoder
        self.head = projection

        loss_kwargs = dict(loss_kwargs)
        self.objective = loss_kwargs.get("objective", "byol")
        self.loss = MMSelfDistillLoss(**loss_kwargs)

        dim = self._head_out_dim(projection)
        pk = dict(predictor_kwargs or {})
        if self.objective == "byol":
            self.predictor = BYOLPredictor(dim, **pk)
        elif self.objective == "dino":
            # DINO has no predictor: the asymmetry is centering + sharpening. The
            # prototype layer is shared, so it must live on both sides and is
            # therefore part of the EMA-copied stack.
            self.predictor = nn.Identity()
            self.head = nn.Sequential(OrderedDict([
                ("proj", projection),
                ("proto", DINOHead(dim, out_dim=loss_kwargs.get("out_dim", 2048), **pk)),
            ]))
        elif self.objective == "jepa":
            # I-JEPA has no projector: the predictor maps context tokens to target
            # token representations directly in encoder space, and the targets are
            # the teacher's LayerNormed tokens. A pooled MLP projector could not be
            # applied here anyway -- its BatchNorm reads the token axis as channels.
            self.head = nn.Identity()
            dim = encoder.fusion_transformer.width
            self.predictor = JEPAPredictor(dim, n_modalities=encoder.num_modalities, **pk)
        else:
            raise ValueError(f"Unknown objective: {self.objective!r}")

        self.mask_kwargs = dict(mode="block", n_targets=4,
                                target_scale=(0.15, 0.2), target_aspect=(0.75, 1.5),
                                context_scale=(0.85, 1.0))
        self.mask_kwargs.update({k: (tuple(v) if isinstance(v, (list, tuple)) else v)
                                 for k, v in (mask_kwargs or {}).items()})
        self.cross_view = (self.objective != "jepa") if cross_view is None else bool(cross_view)

        # the teacher: a frozen copy updated only by EMA
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_head = copy.deepcopy(self.head)
        for p in list(self.teacher_encoder.parameters()) + list(self.teacher_head.parameters()):
            p.requires_grad = False

        self.ema_base = ema_base
        self.ema_final = ema_final
        self._ema_now = ema_base

    @staticmethod
    def _head_out_dim(projection: nn.Module) -> int:
        for m in reversed(list(projection.modules())):
            if isinstance(m, nn.Linear):
                return m.out_features
        raise ValueError("Cannot infer the projector output width")

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def gen_all_possible_masks(self, n_mod: int) -> List[List[bool]]:
        """`n_mod` single-modality masks, then the joint one last."""
        masks = [[s == L for s in range(n_mod)] for L in range(n_mod)]
        masks.append([True] * n_mod)
        return masks

    # ---- teacher ---------------------------------------------------------
    def _ema_momentum(self) -> float:
        # `self.trainer` raises when unattached, so go through the private handle
        trainer = getattr(self, "_trainer", None)
        total = getattr(trainer, "estimated_stepping_batches", None) if trainer else None
        if not total or not math.isfinite(total) or total <= 0:
            return self.ema_base
        t = min(1.0, float(self.global_step) / float(total))
        return self.ema_final - (self.ema_final - self.ema_base) * (math.cos(math.pi * t) + 1) / 2

    @torch.no_grad()
    def update_teacher(self):
        m = self._ema_momentum()
        self._ema_now = m
        for online, target in ((self.encoder, self.teacher_encoder),
                               (self.head, self.teacher_head)):
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(m).add_(po.detach(), alpha=1.0 - m)
            for bo, bt in zip(online.buffers(), target.buffers()):
                bt.copy_(bo)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_teacher()
        self.log("ema_momentum", self._ema_now, on_step=True, sync_dist=False)

    # ---- forward ---------------------------------------------------------
    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]) -> Dict:
        return self._forward_jepa(x1, x2) if self.objective == "jepa" \
            else self._forward_pooled(x1, x2)

    def _forward_pooled(self, x1, x2) -> Dict:
        n_mod = len(x1)
        masks = self.gen_all_possible_masks(n_mod)
        joint = [[True] * n_mod]

        student = []
        for x in (x1, x2):
            z = self.encoder(x, mask_modalities=masks)
            student.append([self.predictor(self.head(zi)) for zi in z])

        teacher = []
        with torch.no_grad():
            for x in (x1, x2):
                z = self.teacher_encoder(x, mask_modalities=joint)
                teacher.append(self.teacher_head(z[0]).detach())

        return {"student": student, "teacher": teacher,
                "cross_view": self.cross_view, "prototype": -1}

    def _forward_jepa(self, x1, x2) -> Dict:
        n_mod = len(x1)
        masks = self.gen_all_possible_masks(n_mod)
        joint = [[True] * n_mod]

        # the teacher sees everything -- no token masking, no modality masking --
        # and the same pass reports the per-modality token counts the split needs
        teacher_tokens, teacher_lengths = [], None
        with torch.no_grad():
            for x in (x1, x2):
                z, lengths = self.teacher_encoder(
                    x, mask_modalities=joint, return_tokens=True, return_token_lengths=True)
                teacher_tokens.append(self.teacher_head(z[0]).detach())
                teacher_lengths = lengths

        splits = sample_ijepa_masks(teacher_lengths, **self.mask_kwargs)
        dev = x1[0].device if isinstance(x1[0], torch.Tensor) else self.device
        ctx_idx = [c.to(dev) for c, _ in splits]
        tgt_blocks = [[t.to(dev) for t in ts] for _, ts in splits]

        offsets, acc = [], 0
        for L in teacher_lengths:
            offsets.append(acc)
            acc += L

        pred, target, joint_by_modality = [], [], []
        for v, x in enumerate((x1, x2)):
            z = self.encoder(x, mask_modalities=masks, return_tokens=True,
                             token_keep=ctx_idx)
            # I-JEPA's target encoder reads the same image as the context encoder;
            # BYOL and DINO are the two-view methods
            tgt_tokens = teacher_tokens[1 - v] if self.cross_view else teacher_tokens[v]
            pv, tv = [], []
            for i, mask in enumerate(masks):
                kept = [m for m, on in enumerate(mask) if on]
                c_mod = torch.cat([torch.full((len(ctx_idx[m]),), m, dtype=torch.long)
                                   for m in kept]).to(dev)
                c_pos = torch.cat([ctx_idx[m] for m in kept])
                branch = kept[0] if len(kept) == 1 else 0
                ctx = self.head(z[i])

                # Every target token is multimodal -- the teacher always sees both
                # modalities, so all of its tokens are contextualised by both. The
                # restriction here is positional, not about target content: a branch
                # is only asked for target positions whose *input stream* is a
                # modality it received. Predicting a position fed by the modality the
                # student never saw would be a cross-modal alignment, which is what
                # CoMM's pair structure exists to avoid.
                #
                # One predictor call per (modality, block), as in I-JEPA: predicting
                # the blocks together would let their queries attend to each other
                # and make the task easier than it is meant to be.
                p_parts, t_parts, spans = [], [], []
                lo = 0
                for m in kept:
                    for blk in tgt_blocks[m]:
                        t_mod = torch.full((len(blk),), m, dtype=torch.long, device=dev)
                        p_parts.append(self.predictor(ctx, c_mod, c_pos, t_mod, blk,
                                                      branch=branch))
                        t_parts.append(tgt_tokens.index_select(1, blk + offsets[m]))
                        lo += len(blk)
                    spans.append(lo)
                pv.append(torch.cat(p_parts, dim=1))
                tv.append(torch.cat(t_parts, dim=1))
                if len(kept) == n_mod and v == 0:
                    # where each modality's targets sit inside the joint branch, so
                    # its error can be compared against that modality's uni branch
                    prev = 0
                    for hi in spans:
                        joint_by_modality.append(torch.arange(prev, hi, device=dev))
                        prev = hi
            pred.append(pv)
            target.append(tv)

        # The context is a block minus every target patch, so its size is not the
        # requested fraction; and it is the knob that decides whether the prediction
        # task has any difficulty, so the achieved value gets logged rather than
        # assumed.
        n_ctx = sum(len(c) for c in ctx_idx)
        keep_achieved = float(n_ctx) / float(max(1, sum(teacher_lengths)))

        return {"pred": pred, "target": target, "keep_achieved": keep_achieved,
                "joint_by_modality": joint_by_modality, "prototype": -1}

    # ---- probing ---------------------------------------------------------
    # Which network each original method evaluates, and why they differ: the probed
    # network is the one that sees the complete input. Both of BYOL's views are full
    # images, so it keeps the online encoder. DINO's student also gets local crops
    # and I-JEPA's context encoder only ever sees a masked subset of patches, so both
    # evaluate the teacher. In this skeleton the modality masks play the same role:
    # the student runs on all M+1 masks, the teacher only on the joint one.
    PROBE_NETWORK = {"byol": "student", "dino": "teacher", "jepa": "teacher"}

    @property
    def probe_encoder(self) -> nn.Module:
        """The encoder the probes read, following each method's own protocol."""
        if self.PROBE_NETWORK[self.objective] == "student":
            return self.encoder
        return self.teacher_encoder

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """Pooled output of `probe_encoder`, never the projector or the predictor."""
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor):
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                output = self.probe_encoder(X_, **kwargs)
                if isinstance(output, list):
                    output = output[0]
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)
