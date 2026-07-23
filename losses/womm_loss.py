import math
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.distributed as dist
from utils import all_gather_batch_with_grad

def all_reduce(tensor, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        reduce_op = dist.ReduceOp.MAX if op == "MAX" else dist.ReduceOp.AVG
        dist.all_reduce(tensor, op=reduce_op)
    return tensor

class WoMMLoss(nn.Module):
    """
        LeJEPA Similarity and Regularization Loss adapted for Multi-Modal Learning
        Maintains the original CoMM data ingestion structure.
    """
    def __init__(self, weights=None, use_rbf=False, sigma_max=2.0, sigma_min=0.5, sigreg_weight=0.05, stop_grad=False):
        super().__init__()
        self.weights = weights
        self.use_rbf = use_rbf
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma = sigma_max
        self.sigreg_weight = sigreg_weight
        self.stop_grad=stop_grad
        
        self.sigreg = SlicingUnivariateTest(EppsPulley(n_points=17), num_slices=4096)

    def step(self, current_epoch, total_epochs):
        if self.use_rbf:
            self.sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * current_epoch / total_epochs))

    def k_sim(self, x, y):
        # dim: [N, D]
        mse = func.mse_loss(x, y, reduction='none').mean(dim=-1) 
        
        if self.use_rbf:
            correntropy = torch.exp(-mse / (2 * self.sigma ** 2))
            return 1.0 - correntropy.mean()
        
        return mse.mean()

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z 
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "loss_sim": torch.Tensor(float), "loss_sigreg": torch.Tensor(float)}
        """
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)
        
        z1 = [z for z in z1]
        z2 = [z for z in z2]
        
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        loss_sim = []
        
        for i in range(n_emb):
            loss1 = self.k_sim(z1[i], z2[prototype].detach() if self.stop_grad else z2[prototype])
            loss2 = self.k_sim(z2[i], z1[prototype].detach() if self.stop_grad else z1[prototype])
            loss_sim.append((loss1 + loss2) / 2.)
            
        losses_dict = {"sim_loss_%i"%i: l for i, l in enumerate(loss_sim)}
        
        if self.weights is not None:
            total_sim_loss = torch.mean(torch.stack(loss_sim) * torch.tensor(self.weights, device=z1[0].device))
        else:
            total_sim_loss = torch.mean(torch.stack(loss_sim))
            
        # dim: [2 * n_emb * N, D]
        z_all_global = torch.cat(Z, dim=0)
        loss_sigreg = self.sigreg(z_all_global)
        
        total_loss = total_sim_loss + (self.sigreg_weight * loss_sigreg)
        
        return {"loss": total_loss, "loss_sim": total_sim_loss, "loss_sigreg": loss_sigreg, **losses_dict}

    def __str__(self):
        return "{}(use_rbf={})".format(type(self).__name__, self.use_rbf)


class SlicingUnivariateTest(torch.nn.Module):
    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        with torch.no_grad():
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            dev = dict(device=x.device)

            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
            
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats


class UnivariateTest(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x):
        return all_reduce(x, op="AVG")

    @property
    def world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1


class EppsPulley(UnivariateTest):
    def __init__(
        self, t_max: float = 5, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt 
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        # dim: (*, N, K, n_points)
        x_t = x.unsqueeze(-1) * self.t 
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # dim: (*, n_points)
        cos_mean = cos_vals.mean(-3) 
        sin_mean = sin_vals.mean(-3) 

        cos_mean = all_reduce(cos_mean, op="AVG")
        sin_mean = all_reduce(sin_mean, op="AVG")

        err = (cos_mean - self.phi).square() + sin_mean.square()

        return (err @ self.weights) * N * self.world_size