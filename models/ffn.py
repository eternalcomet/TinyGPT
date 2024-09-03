from typing import Optional

import torch
from torch import nn, Tensor
from einops import einsum, rearrange


class FFN(nn.Module):

    def __init__(self, d_model: int, is_gated: bool = True, d_mid: Optional[int] = None, tie_kv: bool = False, act_name: str = 'silu'):
        super().__init__()
        self.is_gated = is_gated
        self.d_mid = d_mid
        self.tie_kv = tie_kv
        self.act_name = act_name
        self.d_model = d_model
        
        if d_mid is None:
            # If not specified, make sure the param count is 8 * d ^ 2
            if is_gated:
                d_mid = int(8 * d_model / 3)
            else:
                d_mid = 4 * d_model
        self.w1 = nn.Linear(d_model, d_mid, bias=False)
        self.w2 = nn.Linear(d_mid, d_model, bias=False)
        if is_gated:
            self.w3 = nn.Linear(d_model, d_mid, bias=False)
        
        if tie_kv:
            self.w2.weight.data = self.w1.weight.data.T
            
        self.act_fn = get_act_fn(act_name)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_gated:
            return self.w2(self.act_fn(self.w1(x)) * self.w3(x))
        else:
            return self.w2(self.act_fn(self.w1(x)))


class MHF(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_k: int,
        dim_v: int,
        tie_kv: bool = False,
        d_mid: Optional[int] = None,
        act_name: str = 'silu',
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mid = d_mid
        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.tie_kv = tie_kv
        self.act_name = act_name
        
        if d_mid is None:
            d_mid = d_model * 3
        
        total_dim_k = n_head * dim_k
        total_dim_v = n_head * dim_v
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.Wq = nn.Linear(d_model, total_dim_k, bias=False)
        self.Wo = nn.Linear(total_dim_v, d_model, bias=False)
        if tie_kv:
            assert dim_k == dim_v, f"K and V dimensions must match when tying their weights."
            self.M = nn.Parameter(torch.randn(n_head, d_mid, dim_k) * 0.02)
        else:
            self.K = nn.Parameter(torch.randn(n_head, d_mid, dim_k) * 0.02)
            self.V = nn.Parameter(torch.randn(n_head, d_mid, dim_v) * 0.02)
        
        self.act_fn = get_act_fn(act_name=act_name)
        
    
    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        Q = self.Wq(x)  # (B, T, H * DK)
        Q = rearrange(Q, 'b t (h, dk) -> b t h dk')
        if self.tie_kv:
            w = einsum(Q, self.M, 'b t h dk, h m dk -> b t h m')
            w = self.act_fn(w)
            O = einsum(w, self.M, 'b t h m, h m dv -> b t h dv')
        else:
            w = einsum(Q, self.K, 'b t h dk, h m dk -> b t h m')
            w = self.act_fn(w)
            O = einsum(w, self.V, 'b t h m, h m dv -> b t h dv')
        O = rearrange(O, 'b t h dv -> b t (h dv)')
        y = self.Wo(O)  # (b, t, d)
        return y
