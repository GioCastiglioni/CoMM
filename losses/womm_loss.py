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
    def __init__(
        self, 
        weights=None, 
        reconstruction='mse', 
        regularization='sigreg', 
        temperature=0.1,
        sigma_max=2.0, 
        sigma_min=0.5, 
        reg_weight=0.05, 
        K=4096, 
        stop_grad=False,
        vicreg_inv_coeff=25.0,
        vicreg_std_coeff=25.0,
        vicreg_cov_coeff=1.0
    ):
        super().__init__()
        self.weights = weights
        self.reconstruction = reconstruction
        self.regularization = regularization
        self.temperature = temperature
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma = sigma_max
        self.reg_weight = reg_weight
        self.stop_grad = stop_grad
        self.K = K
        self.INF = 1e8
        
        self.vicreg_std_coeff = vicreg_std_coeff
        self.vicreg_cov_coeff = vicreg_cov_coeff
        self.vicreg_inv_coeff = vicreg_inv_coeff
        
        self._cached_B = -1
        self._cached_target = None
        
        if self.regularization == 'sigreg':
            self.sigreg = SlicingUnivariateTest(EppsPulley(n_points=17), num_slices=self.K)

    def step(self, current_epoch, total_epochs):
        if self.reconstruction == 'rbf':
            self.sigma = self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * current_epoch / total_epochs))

    def calc_similarity(self, x, y):
        # dim: [N, D]
        if self.reconstruction == 'mse':
            return func.mse_loss(x, y, reduction='none').mean(dim=-1)
        elif self.reconstruction == 'l1':
            return func.l1_loss(x, y, reduction='none').mean(dim=-1)
        elif self.reconstruction == 'huber':
            return func.huber_loss(x, y, reduction='none').mean(dim=-1)
        elif self.reconstruction == 'rbf':
            mse = func.mse_loss(x, y, reduction='none').mean(dim=-1)
            correntropy = torch.exp(-mse / (2 * self.sigma ** 2))
            return 1.0 - correntropy
        elif self.reconstruction == 'cosine':
            return - (x * y).sum(dim=-1) / self.temperature
            
        raise ValueError(f"Unknown reconstruction metric: {self.reconstruction}")

    def calc_neg_samples_reg(self, x_norm, y_norm):
        # dim: [N, D]
        N = len(x_norm)
        
        sim_xx = (x_norm @ x_norm.T) / self.temperature
        sim_yy = (y_norm @ y_norm.T) / self.temperature
        sim_xy = (x_norm @ y_norm.T) / self.temperature
        
        sim_xx = sim_xx - self.INF * torch.eye(N, device=x_norm.device)
        sim_yy = sim_yy - self.INF * torch.eye(N, device=x_norm.device)
        
        sim_Z1 = torch.cat([sim_xy, sim_xx], dim=1)
        sim_Z2 = torch.cat([sim_yy, sim_xy.T], dim=1)
        sim_Z = torch.cat([sim_Z1, sim_Z2], dim=0)
        
        return torch.logsumexp(sim_Z, dim=1).mean()

    def calc_semantic_neg_samples_reg(self, x_norm, y_norm, semantic_margin=0.5):
        # dim: [N, D]
        N = len(x_norm)
        
        cos_xx = x_norm @ x_norm.T
        cos_yy = y_norm @ y_norm.T
        cos_xy = x_norm @ y_norm.T
        
        # Stop-gradient semantic prior estimation
        with torch.no_grad():
            prior_xx = torch.clamp(cos_xx, min=0.0)
            prior_yy = torch.clamp(cos_yy, min=0.0)
            prior_xy = torch.clamp(cos_xy, min=0.0)
            
        sim_xx = (cos_xx - semantic_margin * prior_xx) / self.temperature
        sim_yy = (cos_yy - semantic_margin * prior_yy) / self.temperature
        sim_xy = (cos_xy - semantic_margin * prior_xy) / self.temperature
        
        sim_xx = sim_xx - self.INF * torch.eye(N, device=x_norm.device)
        sim_yy = sim_yy - self.INF * torch.eye(N, device=x_norm.device)
        
        # Matrix shape: [2N, 2N]
        sim_Z1 = torch.cat([sim_xy, sim_xx], dim=1)
        sim_Z2 = torch.cat([sim_yy, sim_xy.T], dim=1)
        sim_Z = torch.cat([sim_Z1, sim_Z2], dim=0)
        
        return torch.logsumexp(sim_Z, dim=1).mean()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward_vicreg(self, z: torch.Tensor) -> torch.Tensor:
        # dim: [N, D]
        N, D = z.shape
        z = z - z.mean(dim=0)
        
        std_z = torch.sqrt(z.var(dim=0) + 1e-04)
        std_loss = torch.mean(func.relu(1.0 - std_z))
        
        cov_z = (z.T @ z) / (N - 1)
        cov_loss = self.off_diagonal(cov_z).pow_(2).sum().div(D)
        
        return (self.vicreg_std_coeff * std_loss) + (self.vicreg_cov_coeff * cov_loss)

    def forward_visreg(self, z: torch.Tensor) -> torch.Tensor:
        # dim: [M, B, D]
        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        W = func.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return 0.9 * scale_loss + 1.2 * shape_loss + 0.9 * center_loss

    def _get_target(self, B: int, device) -> torch.Tensor:
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device)

    def forward(self, outputs):
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)

        if self.reconstruction == 'cosine' or self.regularization == 'neg-samples' or self.regularization == 'sem-aware':
            z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
            z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        
        Z = all_gather_batch_with_grad(z1 + z2)
        z1_all, z2_all = Z[:n_emb], Z[n_emb:]

        loss_sim = []
        loss_reg_local = []
        
        for i in range(n_emb):
            tgt1 = z2_all[prototype].detach() if self.stop_grad else z2_all[prototype]
            tgt2 = z1_all[prototype].detach() if self.stop_grad else z1_all[prototype]

            sim1 = self.calc_similarity(z1_all[i], tgt1).mean()
            sim2 = self.calc_similarity(z2_all[i], tgt2).mean()
            loss_sim.append((sim1 + sim2) / 2.0)
            
            if self.regularization == 'neg-samples':
                reg1 = self.calc_neg_samples_reg(z1_all[i], tgt1)
                reg2 = self.calc_neg_samples_reg(z2_all[i], tgt2)
                loss_reg_local.append((reg1 + reg2) / 2.0)

            if self.regularization == 'sem-aware':
                reg1 = self.calc_semantic_neg_samples_reg(z1_all[i], tgt1)
                reg2 = self.calc_semantic_neg_samples_reg(z2_all[i], tgt2)
                loss_reg_local.append((reg1 + reg2) / 2.0)
                
            
        losses_dict = {"sim_loss_%i"%i: l for i, l in enumerate(loss_sim)}
        w_tensor = torch.tensor(self.weights, device=z1_all[0].device) if self.weights is not None else None
        
        if w_tensor is not None:
            total_sim_loss = torch.mean(torch.stack(loss_sim) * w_tensor)
        else:
            total_sim_loss = torch.mean(torch.stack(loss_sim))

        total_sim_loss = (self.vicreg_inv_coeff if self.regularization == 'vicreg' else 1.0) * total_sim_loss
        
        if self.regularization == 'visreg':
            loss_reg = 0.5 * (self.forward_visreg(torch.stack(z1_all, dim=0)) + self.forward_visreg(torch.stack(z2_all, dim=0)))
        elif self.regularization == 'vicreg':
            z1_global = torch.cat(z1_all, dim=0)
            z2_global = torch.cat(z2_all, dim=0)
            loss_reg = 0.5 * (self.forward_vicreg(z1_global) + self.forward_vicreg(z2_global))
        elif self.regularization == 'sigreg':
            z_all_global = torch.cat(Z, dim=0)
            loss_reg = self.sigreg(z_all_global) 
        elif self.regularization == 'neg-samples' or self.regularization == 'sem-aware':
            if w_tensor is not None:
                loss_reg = torch.mean(torch.stack(loss_reg_local) * w_tensor)
            else:
                loss_reg = torch.mean(torch.stack(loss_reg_local))
        else:
            loss_reg = torch.tensor(0.0, device=z1_all[0].device)
            
        total_loss = (1 - self.reg_weight) * total_sim_loss + self.reg_weight * loss_reg
        
        return {"loss": total_loss, "loss_sim": total_sim_loss, "loss_reg": loss_reg, **losses_dict}

    def __str__(self):
        return "{}(rec={}, reg={})".format(type(self).__name__, self.reconstruction, self.regularization)

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