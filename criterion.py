import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import EMA, outer1d, uniform_distributions

# typing only
from torch import Tensor
from torch.types import Device


@torch.no_grad()
def _semisup_logits_weights(y: Tensor, label_weights: Tensor) -> Tensor:
    neg_weights = torch.where(
        outer1d(y, y, op=torch.eq),
        1 - outer1d(label_weights, label_weights),
        1
    )
    neg_weights.fill_diagonal_(1)

    weights = neg_weights.repeat(2, 2)
    weights.fill_diagonal_(0)

    return weights


class SoftSemiSupInfoNCELoss(nn.Module):

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_w: Tensor, z_s: Tensor, y: Tensor, label_weights: Tensor) -> Tensor:
        z = torch.cat([z_w, z_s])
        logits = (z @ z.T) / self.temperature

        weights = _semisup_logits_weights(y, label_weights)
        logits += torch.log(weights)

        batch_size = len(y)
        target = torch.arange(2 * batch_size, device=y.device).roll(batch_size)

        return F.cross_entropy(logits, target)


ContrastiveLoss = SoftSemiSupInfoNCELoss


class SoftMatchLossBase(nn.Module):

    def __init__(
        self,
        n_clusters: int,
        outlier_loss_weight: float = 1.,
        momentum: float = 0.99,
        use_ua: bool = True,
        label_smoothing: float = 0.1,
        estimate_outlier_only: bool = False,
        initial_step: int = 0,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.outlier_loss_weight = outlier_loss_weight
        self.m = momentum
        self.use_ua = use_ua
        self.label_smoothing = label_smoothing
        self.estimate_outlier_only = estimate_outlier_only
        self.step = initial_step

        self.reset_statistics()
    
    def reset_statistics(self) -> None:
        self.p_mean_ema = EMA(self.m, 0.0)
        self.p_max_mean_ema = EMA(self.m, 1 / self.n_clusters)
        self.p_max_var_ema = EMA(self.m, None)
    
    @torch.no_grad()
    def uniform_align(self, p: Tensor) -> Tensor:
        if self.training:
            p_mean = p.mean(dim=0)
            p_mean = self.p_mean_ema(p_mean)
        else:
            p_mean = self.p_mean_ema.value
        
        p = p * (1 / self.n_clusters) / p_mean
        p = p / p.sum(dim=-1, keepdim=True)

        return p
    
    @torch.no_grad()
    def outlier_weights(self, p: Tensor) -> Tensor:
        p = p.detach()

        if self.use_ua:
            p = self.uniform_align(p)

        p_max = p.amax(dim=1)
        
        if self.training:
            p_max_var, p_max_mean = torch.var_mean(p_max, unbiased=True)
            p_max_mean = self.p_max_mean_ema(p_max_mean.item())
            p_max_var = self.p_max_var_ema(p_max_var.item())
        else:
            p_max_mean = self.p_max_mean_ema.value
            p_max_var = self.p_max_var_ema.value

        return torch.exp(
            -torch.clamp_max(p_max - p_max_mean, 0.0).square()
            / (2 * p_max_var)
        )


class SoftClusteringLoss(SoftMatchLossBase):

    def forward(
        self,
        p: Tensor,
        p_w: Tensor,
        p_s: Tensor,
        y: Tensor,
        label_weights: Tensor
    ) -> tuple[Tensor, Tensor]:
        loss_w_sup = label_weights * F.cross_entropy(
            p_w, y, reduction='none',
            label_smoothing=self.label_smoothing,
        )
        loss_s_sup = label_weights * F.cross_entropy(
            p_s, y, reduction='none',
            label_smoothing=self.label_smoothing,
        )
        loss_sup = loss_w_sup.mean() + loss_s_sup.mean()
        
        p_w = p_w.softmax(dim=1)
        weights = self.outlier_weights(p_w)
        y_pseudo = p_w.argmax(dim=-1).detach()

        loss_unsup = weights * F.cross_entropy(
            p_s, y_pseudo, reduction='none',
            label_smoothing=self.label_smoothing,
        )
        loss_unsup = self.outlier_loss_weight * loss_unsup.mean()

        # loss_unsup = loss_sup.new_zeros((), requires_grad=True)

        if self.training:
            self.step += 1

        return loss_sup, loss_unsup


ClusteringLoss = SoftClusteringLoss
