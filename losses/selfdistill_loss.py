"""The three self-distillation objectives behind one interface.

Every objective sees the same pairs as CoMM: mask `m` of one view against the
*joint* embedding of the other view. The only change from `WoMMLoss` is that the
joint side comes from an EMA teacher instead of the online network, and that there
is no explicit regularisation term -- collapse is meant to be prevented by the
teacher/student asymmetry alone. That is what makes this family a different point
in the design space rather than another regulariser.
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class MMSelfDistillLoss(nn.Module):
    OBJECTIVES = ("byol", "dino", "jepa")

    def __init__(self,
                 objective: str = "byol",
                 weights: Optional[List[float]] = None,
                 # dino
                 student_temp: float = 0.1,
                 teacher_temp: float = 0.07,
                 warmup_teacher_temp: float = 0.04,
                 warmup_teacher_temp_frac: float = 0.3,
                 center_momentum: float = 0.9,
                 out_dim: int = 2048,
                 # jepa
                 huber_delta: float = 1.0):
        super().__init__()
        if objective not in self.OBJECTIVES:
            raise ValueError(f"Unknown objective {objective!r}, expected one of {self.OBJECTIVES}")
        self.objective = objective
        self.weights = weights
        self.student_temp = student_temp
        self.teacher_temp_final = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_frac = warmup_teacher_temp_frac
        self.center_momentum = center_momentum
        self.huber_delta = huber_delta
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("teacher_temp", torch.tensor(float(warmup_teacher_temp)))

    # ---- schedules -------------------------------------------------------
    def step(self, epoch: int, max_epochs: int):
        """Linear warmup of the teacher temperature, as in DINO.

        Starting sharp makes the teacher confident before it is any good, which is
        the reported route to collapse in DINO.
        """
        if self.objective != "dino":
            return
        n = max(1, int(round(max_epochs * self.warmup_teacher_temp_frac)))
        t = min(1.0, float(epoch) / n)
        self.teacher_temp.fill_(self.warmup_teacher_temp
                                + t * (self.teacher_temp_final - self.warmup_teacher_temp))

    # ---- per-objective discrepancies -------------------------------------
    def _byol(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = F.normalize(student, dim=-1, p=2)
        t = F.normalize(teacher, dim=-1, p=2)
        return (2.0 - 2.0 * (s * t).sum(dim=-1)).mean()

    def _dino(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        t = F.softmax((teacher - self.center) / self.teacher_temp, dim=-1)
        s = F.log_softmax(student / self.student_temp, dim=-1)
        return -(t * s).sum(dim=-1).mean()

    def _jepa(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.huber_delta)

    @torch.no_grad()
    def update_center(self, teacher: torch.Tensor):
        if self.objective != "dino":
            return
        batch_center = teacher.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum)

    # ---- diagnostics -----------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _effective_rank(z: torch.Tensor) -> torch.Tensor:
        """exp(entropy of the normalised singular values).

        The collapse metric for this family: there is no regularisation term whose
        value would reveal a shrinking representation, so it has to be measured.
        """
        z = z.float()
        z = z - z.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(z)
        p = s / s.sum().clamp_min(1e-12)
        p = p.clamp_min(1e-12)
        return torch.exp(-(p * p.log()).sum())

    # ---- forward ---------------------------------------------------------
    def forward(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        if self.objective == "jepa":
            return self._forward_jepa(outputs)
        return self._forward_pooled(outputs)

    def _weights(self, n: int, device) -> Optional[torch.Tensor]:
        if self.weights is None:
            return None
        if len(self.weights) != n:
            raise ValueError(f"`weights` has {len(self.weights)} entries for {n} masks")
        return torch.tensor(self.weights, device=device, dtype=torch.float32)

    def _forward_pooled(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        student, teacher = outputs["student"], outputs["teacher"]
        n_mask = len(student[0])
        d = self._byol if self.objective == "byol" else self._dino
        # DINO's rule that a student crop is never matched to the teacher's view of
        # the same crop is what `cross_view` implements here
        cv = bool(outputs.get("cross_view", True))
        per_mask = []
        for i in range(n_mask):
            per_mask.append(0.5 * (d(student[0][i], teacher[1 if cv else 0])
                                   + d(student[1][i], teacher[0 if cv else 1])))
        per_mask = torch.stack(per_mask)
        w = self._weights(n_mask, per_mask.device)
        loss = torch.mean(per_mask * w) if w is not None else per_mask.mean()

        out = {"loss": loss}
        out.update({f"sd_loss_{i}": v for i, v in enumerate(per_mask)})
        with torch.no_grad():
            out["teacher_rank"] = self._effective_rank(teacher[0])
            out["student_rank"] = self._effective_rank(student[0][-1])
            if self.objective == "dino":
                p = F.softmax((teacher[0] - self.center) / self.teacher_temp, dim=-1)
                out["teacher_entropy"] = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
                out["teacher_pmax"] = p.max(dim=-1).values.mean()
                out["teacher_temp_now"] = self.teacher_temp.clone()
        self.update_center(torch.cat([teacher[0], teacher[1]], dim=0))
        return out

    def _forward_jepa(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        pred, target = outputs["pred"], outputs["target"]
        n_mask = len(pred[0])
        per_mask = []
        for i in range(n_mask):
            per_mask.append(0.5 * (self._jepa(pred[0][i], target[0][i])
                                   + self._jepa(pred[1][i], target[1][i])))
        per_mask = torch.stack(per_mask)
        w = self._weights(n_mask, per_mask.device)
        loss = torch.mean(per_mask * w) if w is not None else per_mask.mean()

        out = {"loss": loss}
        out.update({f"sd_loss_{i}": v for i, v in enumerate(per_mask)})
        if "keep_achieved" in outputs:
            out["keep_achieved"] = torch.as_tensor(
                outputs["keep_achieved"], device=loss.device, dtype=loss.dtype)
        with torch.no_grad():
            # the joint branch is last; its targets are a superset of every uni branch's
            joint = target[0][-1]
            out["teacher_rank"] = self._effective_rank(joint.reshape(-1, joint.shape[-1]))
            # how much the other modality helped, per modality, on shared targets
            for m, split in enumerate(outputs.get("joint_by_modality", [])):
                if split is not None and split.numel() > 0:
                    out[f"joint_on_mod{m + 1}"] = self._jepa(
                        pred[0][-1][:, split], target[0][-1][:, split])
        return out
