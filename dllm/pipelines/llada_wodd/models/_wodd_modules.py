"""
WoDD building blocks.

Three modules:
  * WriteHead       — produce (gate g_k, content v_k) from pooled hidden state
  * TapeRead        — cross-attention from sequence positions into the tape
  * MemoryTape      — append-only buffer holding [g_1 * v_1, ..., g_K * v_K]

Design constraints
------------------
1. Identity at init: when zero_init_wodd=True, both WriteHead.content_proj and
   TapeRead.out_proj are zero-initialized so the model at step 0 is bit-
   identical to vanilla LLaDA with the same prelude/recurrent/coda split.
2. No discrete sampling: g_k is a sigmoid scalar in [0, 1]; we multiply it
   onto v_k BEFORE append. Tape entries with g_k ~ 0 contribute ~0 to cross-
   attention via the multiplicative gate, recovering "append-only with skips"
   semantics without ST-Gumbel.
3. Append-only: MemoryTape only supports .append() and .stack(); no in-place
   overwrite. This is the formal claim WoDD makes vs MetaState-style
   evolution-in-place.
4. Stateless training: the tape lives for one forward pass only. Across
   training samples / forward calls it is freshly initialized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pooling helper
# ---------------------------------------------------------------------------

def masked_mean(
    h: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Mean over the token axis, respecting attention_mask if provided.

    Args:
        h: (B, T, D)
        attention_mask: (B, T) with 1 for real tokens, 0 for pad, or None.

    Returns:
        (B, D)
    """
    if attention_mask is None:
        return h.mean(dim=1)
    mask = attention_mask.to(h.dtype).unsqueeze(-1)  # (B, T, 1)
    summed = (h * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


# ---------------------------------------------------------------------------
# WriteHead
# ---------------------------------------------------------------------------

class WriteHead(nn.Module):
    """Decide whether and what to write to the tape this inner step.

    Forward:
        pool: (B, d_model)  -- pooled hidden state of visible tokens
        -> g: (B, 1)        sigmoid gate, range [0, 1]
        -> v: (B, tape_dim) content vector

    Tape entry that ultimately gets appended is `g * v`, so g ~ 0 means a
    near-zero contribution to subsequent cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        tape_dim: int,
        zero_init: bool = True,
        gate_bias_init: float = -2.0,
        init_device: Optional[str] = None,
    ):
        super().__init__()
        # We bottleneck via tape_dim because pooled hidden is high-dim but the
        # signal we care about is "what to remember", which is lower-rank.
        self.pre_ln = nn.LayerNorm(d_model, device=init_device)
        self.gate_proj = nn.Linear(d_model, 1, device=init_device)
        self.content_proj = nn.Linear(d_model, tape_dim, device=init_device)

        # gate_bias_init=-2.0 => sigmoid(-2) ~ 0.12 at init, so the model
        # writes sparsely from the start (anti-collapse to always-write).
        # The Universal Transformers Need Memory paper recommends bias=-3 for
        # similar ACT-style halting -- we use -2 as a softer start; sparsity
        # regularizer will push down further if needed.
        with torch.no_grad():
            self.gate_proj.bias.fill_(float(gate_bias_init))

        if zero_init:
            # Zero-init content output so v_k = 0 at init -> tape stores all-
            # zero entries -> cross-attention output is 0 -> model is exactly
            # vanilla LLaDA at training step 0.
            with torch.no_grad():
                self.content_proj.weight.zero_()
                if self.content_proj.bias is not None:
                    self.content_proj.bias.zero_()

    def forward(
        self,
        pool: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pre_ln(pool)
        g_logit = self.gate_proj(x)             # (B, 1)
        g = torch.sigmoid(g_logit)
        v = self.content_proj(x)                # (B, tape_dim)
        return g, v


# ---------------------------------------------------------------------------
# MemoryTape
# ---------------------------------------------------------------------------

class MemoryTape:
    """Append-only buffer of tape entries for one forward pass.

    Stores a list of (B, tape_dim) tensors. Provides .stack() to materialize
    them as (B, K, tape_dim) for cross-attention.

    No in-place updates. No overwrites. This is the formal append-only
    semantic distinguishing WoDD from state-evolution methods.
    """

    def __init__(self, batch_size: int, tape_dim: int, max_writes: int):
        self.batch_size = int(batch_size)
        self.tape_dim = int(tape_dim)
        self.max_writes = int(max_writes)
        self._entries: List[torch.Tensor] = []
        self._gates: List[torch.Tensor] = []   # for diagnostics

    def append(self, gated_content: torch.Tensor, gate: torch.Tensor) -> None:
        assert gated_content.shape == (self.batch_size, self.tape_dim), (
            f"Expected ({self.batch_size}, {self.tape_dim}), "
            f"got {tuple(gated_content.shape)}"
        )
        if len(self._entries) >= self.max_writes:
            raise RuntimeError(
                f"MemoryTape full: cannot append entry {len(self._entries) + 1} "
                f"(max_writes={self.max_writes})"
            )
        self._entries.append(gated_content)
        self._gates.append(gate)

    def __len__(self) -> int:
        return len(self._entries)

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def stack(self) -> torch.Tensor:
        """(B, K, tape_dim) — current tape contents.

        Raises if empty; callers should check `is_empty()` first.
        """
        if not self._entries:
            raise RuntimeError("MemoryTape is empty; call .is_empty() first")
        return torch.stack(self._entries, dim=1)

    def gates(self) -> torch.Tensor:
        """(B, K, 1) — gate values for every write (for sparsity loss)."""
        if not self._gates:
            # Return a degenerate tensor with the right shape for downstream code.
            # We don't know the device here; the caller has to provide an empty
            # tensor manually if needed. In practice, write_gates is always non-
            # empty after the WoDD forward loop runs T_step >= 1 times.
            raise RuntimeError("MemoryTape gates empty")
        return torch.stack(self._gates, dim=1)


# ---------------------------------------------------------------------------
# TapeRead — multi-head cross-attention from sequence into tape
# ---------------------------------------------------------------------------

class TapeRead(nn.Module):
    """Cross-attention block: sequence positions attend to tape entries.

    Q: from h (the main hidden state, B, T, d_model)
    K, V: from tape entries (B, K, tape_dim)

    Output: (B, T, d_model), added back to h via residual outside this module.

    Notes:
      - Output projection zero-init (when enabled) makes step-0 forward
        bit-identical to vanilla split LLaDA.
      - Position info on tape entries comes from a learned write-index
        embedding added to K before projection -- so the model can tell
        "first write" from "last write".
    """

    def __init__(
        self,
        d_model: int,
        tape_dim: int,
        max_writes: int,
        n_heads: int = 8,
        zero_init: bool = True,
        init_device: Optional[str] = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.tape_dim = int(tape_dim)
        self.max_writes = int(max_writes)
        self.n_heads = int(n_heads)
        assert tape_dim % n_heads == 0
        self.head_dim = tape_dim // n_heads

        self.q_ln = nn.LayerNorm(d_model, device=init_device)
        self.kv_ln = nn.LayerNorm(tape_dim, device=init_device)

        self.q_proj = nn.Linear(d_model, tape_dim, device=init_device)
        self.k_proj = nn.Linear(tape_dim, tape_dim, device=init_device)
        self.v_proj = nn.Linear(tape_dim, tape_dim, device=init_device)
        self.out_proj = nn.Linear(tape_dim, d_model, device=init_device)

        # Write-index positional embedding. Distinct vector for each possible
        # write slot up to max_writes.
        self.write_pos_emb = nn.Embedding(
            max_writes, tape_dim, device=init_device
        )

        if zero_init:
            with torch.no_grad():
                self.out_proj.weight.zero_()
                if self.out_proj.bias is not None:
                    self.out_proj.bias.zero_()

    def forward(
        self,
        h: torch.Tensor,        # (B, T, d_model)
        tape: torch.Tensor,     # (B, K, tape_dim)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:
        B, T, _ = h.shape
        _, K, _ = tape.shape

        # Add write-index positional embedding to tape entries.
        pos_ids = torch.arange(K, device=tape.device)
        tape = tape + self.write_pos_emb(pos_ids).unsqueeze(0)  # broadcast over B

        q = self.q_proj(self.q_ln(h))            # (B, T, tape_dim)
        kv = self.kv_ln(tape)
        k = self.k_proj(kv)                      # (B, K, tape_dim)
        v = self.v_proj(kv)                      # (B, K, tape_dim)

        # Reshape for multi-head: (B, n_heads, *, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        # Standard scaled dot-product attention.
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # No causal mask: tape has no temporal order constraint; the write-pos
        # embedding is the only positional info. No padding mask on K because
        # absent writes are simply not in the tape.
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)              # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.tape_dim)
        out = self.out_proj(out)                 # (B, T, d_model)

        # Optionally zero out output at padded positions to keep gradients
        # clean. The downstream loss already masks pad tokens, so this is
        # mostly cosmetic.
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)

        return out


# ---------------------------------------------------------------------------
# Aggregate stats helpers (for loss / logging)
# ---------------------------------------------------------------------------

@dataclass
class WoDDStepStats:
    write_gates: torch.Tensor       # (B, T_step, 1)  — all g_k in this forward

    @property
    def mean_gate(self) -> torch.Tensor:
        """Per-sample expected write rate, in [0, 1]. (B,)"""
        return self.write_gates.squeeze(-1).mean(dim=-1)

    @property
    def gate_entropy(self) -> torch.Tensor:
        """Bernoulli entropy of every gate, averaged.

        H(p) = -p log p - (1-p) log(1-p). We want this NOT to collapse to 0,
        so the entropy regularizer is added with a *negative* coefficient
        (i.e. encourages entropy).
        """
        p = self.write_gates.squeeze(-1).clamp(min=1e-6, max=1.0 - 1e-6)
        h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return h.mean(dim=-1)  # (B,)
