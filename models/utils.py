import torch.nn.functional as F
import torch
from torch import nn, Tensor


def get_act_fn(act_name: str):
    if act_name == 'relu':
        return F.relu
    elif act_name == 'silu':
        return F.silu
    elif act_name == 'gelu':
        return F.gelu
    else:
        raise ValueError


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
