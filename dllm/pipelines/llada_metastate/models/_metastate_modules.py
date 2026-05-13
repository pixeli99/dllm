"""Helper modules for LLaDA-MetaState.

Three modules:
  - `StateRead`  : sequence-to-state cross-attention (analogue of WoDD's
    TapeRead but with no positional embedding -- MetaState slots have no
    intrinsic order beyond what the GRU update bakes in).
  - `MetaStateGRU` : a slot-parallel GRU update that maps
      (state[m] in R^D, pool in R^d_model) -> state[m]'  with the same
    reset/update/new gate structure as `torch.nn.GRUCell` but batched
    over the M slot axis.
  - `masked_mean`  : duplicated from WoDD's _wodd_modules.py to keep the
    module self-contained (no cross-pipeline imports).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(h: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over the token axis (dim=1), respecting attention_mask."""
    if attention_mask is None:
        return h.mean(dim=1)
    am = attention_mask.to(h.dtype).unsqueeze(-1)        # (B, T, 1)
    denom = am.sum(dim=1).clamp_min(1e-6)                # (B, 1)
    return (h * am).sum(dim=1) / denom                   # (B, d_model)


# ---------------------------------------------------------------------------
# State read: sequence -> state cross-attention
# ---------------------------------------------------------------------------

class StateRead(nn.Module):
    """Cross-attention block: token positions attend to MetaState slots.

    Q from h (B, T, d_model); K, V from state (B, M, D). Output (B, T,
    d_model) is added to h via residual outside this module.

    With `zero_init=True`, out_proj is zero-initialized so that the
    cross-attention contribution to h is exactly 0 at iteration 0 of
    iteration 0 of training (matched-FLOPs initialisation).
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        n_heads: int = 8,
        zero_init: bool = True,
        init_device: Optional[str] = None,
    ):
        super().__init__()
        self.d_model   = int(d_model)
        self.state_dim = int(state_dim)
        self.n_heads   = int(n_heads)
        assert state_dim % n_heads == 0
        self.head_dim  = state_dim // n_heads

        self.q_ln  = nn.LayerNorm(d_model,   device=init_device)
        self.kv_ln = nn.LayerNorm(state_dim, device=init_device)
        self.q_proj   = nn.Linear(d_model,   state_dim, device=init_device)
        self.k_proj   = nn.Linear(state_dim, state_dim, device=init_device)
        self.v_proj   = nn.Linear(state_dim, state_dim, device=init_device)
        self.out_proj = nn.Linear(state_dim, d_model,   device=init_device)

        if zero_init:
            with torch.no_grad():
                self.out_proj.weight.zero_()
                if self.out_proj.bias is not None:
                    self.out_proj.bias.zero_()

    def forward(
        self,
        h: torch.Tensor,         # (B, T, d_model)
        state: torch.Tensor,     # (B, M, state_dim)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:
        B, T, _ = h.shape
        _, M, _ = state.shape

        q  = self.q_proj(self.q_ln(h))                       # (B, T, D)
        kv = self.kv_ln(state)
        k  = self.k_proj(kv)                                  # (B, M, D)
        v  = self.v_proj(kv)                                  # (B, M, D)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, v)                          # (B, H, T, head_dim)
        out  = out.transpose(1, 2).contiguous().view(B, T, self.state_dim)
        out  = self.out_proj(out)                             # (B, T, d_model)

        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        return out


# ---------------------------------------------------------------------------
# State update: slot-parallel GRU
# ---------------------------------------------------------------------------

class MetaStateGRU(nn.Module):
    """Per-slot GRU update: state[m] <- GRU(state[m], pool).

    The GRU treats `pool` (the pooled hidden state of the sequence) as the
    input and `state[m]` as the previous hidden state. All M slots are
    updated in parallel via a (B, M, D) tensor.

    Standard GRU gates:
        r = sigmoid(W_xr pool + W_hr state + b_r)
        z = sigmoid(W_xz pool + W_hz state + b_z)
        n = tanh   (W_xn pool + r * (W_hn state + b_hn) + b_xn)
        state' = (1 - z) * state + z * n

    With `zero_init=True` we initialise the update gate's bias to a large
    negative number so `z ~ 0` at init, i.e. state stays put. Combined
    with StateRead's zero out_proj, the model at init is bit-identical
    to a vanilla LLaDA forward (no information flows through the state).

    Trainable param count: O(d_model * D + D * D), which is independent
    of M -- this is what lets Phase 3 vary `state_slots` while holding
    trainable parameters roughly constant.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        zero_init: bool = True,
        update_bias_init: float = -5.0,
        init_device: Optional[str] = None,
    ):
        super().__init__()
        D = int(state_dim)
        # Shared linear layers operate slot-parallel (state has shape (B, M, D)).
        self.W_xr = nn.Linear(d_model, D, bias=True,  device=init_device)
        self.W_hr = nn.Linear(D,       D, bias=False, device=init_device)
        self.W_xz = nn.Linear(d_model, D, bias=True,  device=init_device)
        self.W_hz = nn.Linear(D,       D, bias=False, device=init_device)
        self.W_xn = nn.Linear(d_model, D, bias=True,  device=init_device)
        self.W_hn = nn.Linear(D,       D, bias=True,  device=init_device)

        if zero_init:
            # sigmoid(-5) ~ 6.7e-3 so the update gate is nearly closed at
            # init; state[m]' ~ state[m]. Combined with StateRead.out_proj=0
            # we get the strict drop-in invariant.
            with torch.no_grad():
                self.W_xz.bias.fill_(float(update_bias_init))

    def forward(
        self,
        state: torch.Tensor,    # (B, M, D)
        pool: torch.Tensor,     # (B, d_model)
    ) -> torch.Tensor:
        # Project pool once; broadcast across M.
        xr = self.W_xr(pool).unsqueeze(1)   # (B, 1, D)
        xz = self.W_xz(pool).unsqueeze(1)
        xn = self.W_xn(pool).unsqueeze(1)

        r = torch.sigmoid(xr + self.W_hr(state))
        z = torch.sigmoid(xz + self.W_hz(state))
        n = torch.tanh   (xn + r * self.W_hn(state))
        return (1.0 - z) * state + z * n


# ---------------------------------------------------------------------------
# MetaState init buffer
# ---------------------------------------------------------------------------

class MetaStateInit(nn.Module):
    """Learnable initial state buffer of shape (M, D).

    At forward time we broadcast to (B, M, D). The state_init bias is
    NOT counted toward the "trainable parameter budget" comparison in
    Phase 3 because it grows linearly in M (state_slots) by design.
    """

    def __init__(
        self,
        state_slots: int,
        state_dim: int,
        zero_init: bool = True,
        init_device: Optional[str] = None,
    ):
        super().__init__()
        self.state_slots = int(state_slots)
        self.state_dim   = int(state_dim)
        # Use a parameter (not a Buffer) so it learns; std=0.02 is the
        # standard transformer init when zero_init=False.
        init = torch.zeros(self.state_slots, self.state_dim, device=init_device)
        if not zero_init:
            init.normal_(mean=0.0, std=0.02)
        self.state = nn.Parameter(init)

    def forward(self, B: int) -> torch.Tensor:
        return self.state.unsqueeze(0).expand(B, -1, -1).contiguous()
