from typing import Optional

import torch
from torch import nn, Tensor
from einops import einsum, rearrange

from .utils import get_act_fn, RMSNorm


class FFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        is_gated: bool = True,
        d_mid: Optional[int] = None,
        tie_kv: bool = False,
        act_name: str = "silu",
    ):
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
    """
    A multi-headed version of FFN, keys and values are learnable neural weights, while the input
    is mapped to a multi-headed query. Each head operates on a subset of dimensions, and
    the heads are computed in parallel.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_k: int,
        dim_v: int,
        tie_kv: bool = False,
        d_mid: Optional[int] = None,
        use_proj_q: bool = True,
        use_proj_o: bool = True,
        q_norm: bool = True,
        act_name: str = "silu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mid = d_mid
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.tie_kv = tie_kv
        self.act_name = act_name

        if d_mid is None:
            if use_proj_q and use_proj_o:
                d_mid = d_model * 3
            elif use_proj_q or use_proj_o:
                d_mid = int(d_model * 3.5)
            else:
                d_mid = d_model * 4

        total_dim_k = n_heads * dim_k
        total_dim_v = n_heads * dim_v
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.use_proj_q = use_proj_q
        self.use_proj_o = use_proj_o
        self.q_norm = q_norm

        if use_proj_q:
            self.Wq = nn.Linear(d_model, total_dim_k, bias=False)
        else:
            assert total_dim_k == d_model

        if use_proj_o:
            self.Wo = nn.Linear(total_dim_v, d_model, bias=False)
        else:
            assert total_dim_v == d_model

        if tie_kv:
            assert (
                dim_k == dim_v
            ), f"K and V dimensions must match when tying their weights, got {dim_k=}, {dim_v=}"
            self.M = nn.Parameter(torch.zeros(n_heads, d_mid, dim_k))
            torch.nn.init.normal_(self.M.data, std=0.02)
        else:
            self.K = nn.Parameter(torch.zeros(n_heads, d_mid, dim_k))
            self.V = nn.Parameter(torch.zeros(n_heads, d_mid, dim_v))
            torch.nn.init.normal_(self.K.data, std=0.02)
            torch.nn.init.normal_(self.V.data, std=0.02)

        self.act_fn = get_act_fn(act_name=act_name)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seqlen, d_model = x.shape
        if self.use_proj_q:
            Q = self.Wq(x)  # (B, T, H * DK)
        else:
            Q = x
        Q = rearrange(Q, "b t (h dk) -> b t h dk", h=self.n_heads)
        if self.tie_kv:
            scores = einsum(Q, self.M, "b t h dk, h m dk -> b t h m")
            scores = self.act_fn(scores)
            out = einsum(scores, self.M, "b t h m, h m dv -> b t h dv")
        else:
            scores = einsum(Q, self.K, "b t h dk, h m dk -> b t h m")
            scores = self.act_fn(scores)
            out = einsum(scores, self.V, "b t h m, h m dv -> b t h dv")
        out = rearrange(out, "b t h dv -> b t (h dv)")
        if self.use_proj_o:
            y = self.Wo(out)  # (b, t, d)
        else:
            y = out
        return y
