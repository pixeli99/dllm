"""Determinism tests for the trajectory-recording sampler.

How to run::

    cd /Users/pixeli/dllm
    python -m pytest tests/td/test_sampler_deterministic.py -v

These tests use a tiny stub LLM on CPU; no real LLaDA, no GPU. We only
test sampler-level determinism + tie-break behaviour. Numerical parity
between micro-batch sizes (which depends on the underlying attention
kernel) is not in scope here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from dllm.core.samplers.mdlm_deterministic import (
    DeterministicMDLMSampler,
    DeterministicMDLMSamplerConfig,
    sampler_config_hash_payload,
)


# ---------------------------------------------------------------------------
# Stubs (no real model, fast on CPU)
# ---------------------------------------------------------------------------


class _StubLM(nn.Module):
    """A tiny token-conditioned linear head -- deterministic given inputs."""

    def __init__(self, vocab_size: int, hidden_dim: int = 32, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        h = self.embed(input_ids)  # [B, T, D]
        logits = self.head(h)  # [B, T, V]
        return SimpleNamespace(logits=logits)


class _StubTokenizer:
    """Just enough surface for DeterministicMDLMSampler."""

    def __init__(self, vocab_size: int, mask_id: int, eos_id: int, bos_id: int):
        self.mask_token_id = mask_id
        self.eos_token_id = eos_id
        self.bos_token_id = bos_id
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size


def _make_sampler(seed: int = 0, vocab_size: int = 200) -> DeterministicMDLMSampler:
    mask_id = vocab_size - 1
    eos_id = vocab_size - 2
    bos_id = vocab_size - 3
    model = _StubLM(vocab_size=vocab_size, seed=seed).eval()
    for p in model.parameters():
        p.requires_grad = False
    tok = _StubTokenizer(vocab_size, mask_id, eos_id, bos_id)
    return DeterministicMDLMSampler(model=model, tokenizer=tok)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_sample_with_trajectory_bit_exact_repeat():
    """Same prompts + same config + same model -> bit-exact sequences AND trajectories."""
    sampler = _make_sampler()
    cfg = DeterministicMDLMSamplerConfig(
        teacher_steps=8, max_response_len=8, top_k=4
    )
    prompts = [
        torch.tensor([5, 6, 7, 8], dtype=torch.long),
        torch.tensor([10, 11, 12], dtype=torch.long),
    ]

    out1 = sampler.sample_with_trajectory(prompts, cfg)
    out2 = sampler.sample_with_trajectory(prompts, cfg)

    # Final sequences identical bit-for-bit
    torch.testing.assert_close(out1.sequences, out2.sequences, rtol=0, atol=0)
    assert out1.prompt_lens == out2.prompt_lens

    # Trajectory entries identical bit-for-bit
    assert len(out1.trajectories) == len(out2.trajectories)
    for traj1, traj2 in zip(out1.trajectories, out2.trajectories):
        assert len(traj1) == len(traj2)
        for s1, s2 in zip(traj1, traj2):
            assert s1.step_idx == s2.step_idx
            np.testing.assert_array_equal(s1.mask_state, s2.mask_state)
            np.testing.assert_array_equal(s1.top_k_indices, s2.top_k_indices)
            # fp32 should be exactly equal across two runs of the same model
            np.testing.assert_array_equal(s1.top_k_logprobs, s2.top_k_logprobs)
            np.testing.assert_array_equal(s1.neg_log_tail_mass, s2.neg_log_tail_mass)
            np.testing.assert_array_equal(s1.top1_conf, s2.top1_conf)
            np.testing.assert_array_equal(s1.entropy, s2.entropy)


def test_trajectory_step_count_matches_teacher_steps():
    sampler = _make_sampler()
    L_resp = 16
    cfg = DeterministicMDLMSamplerConfig(
        teacher_steps=L_resp, max_response_len=L_resp, top_k=8
    )
    prompts = [torch.tensor([1, 2, 3], dtype=torch.long)]
    out = sampler.sample_with_trajectory(prompts, cfg)
    assert len(out.trajectories[0]) == L_resp


def test_response_fully_unmasked_at_end():
    sampler = _make_sampler()
    cfg = DeterministicMDLMSamplerConfig(
        teacher_steps=8, max_response_len=8, top_k=4
    )
    prompts = [torch.tensor([1, 2, 3], dtype=torch.long)]
    out = sampler.sample_with_trajectory(prompts, cfg)

    pl = out.prompt_lens[0]
    response = out.sequences[0, pl : pl + cfg.max_response_len]
    mask_id = sampler.tokenizer.mask_token_id
    assert (response != mask_id).all().item(), "response still has mask tokens"


# ---------------------------------------------------------------------------
# Tie-break behaviour
# ---------------------------------------------------------------------------


class _ConstLogitLM(nn.Module):
    """Returns the SAME logits at every position -- forces ties everywhere.

    Output logits are sample-and-position-independent: ``logits[b, t, v] = const[v]``.
    With this, the per-position confidence (max softmax) is identical at
    every position, so the sampler must pick by tie-break alone.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        # Bias the favorite token slightly so argmax is well-defined.
        self.const_logits = nn.Parameter(torch.zeros(vocab_size), requires_grad=False)
        with torch.no_grad():
            self.const_logits[0] = 5.0  # everyone agrees: token 0 wins

    @property
    def device(self):
        return self.const_logits.device

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        B, T = input_ids.shape
        logits = self.const_logits.view(1, 1, -1).expand(B, T, -1).contiguous()
        return SimpleNamespace(logits=logits)


def test_tie_break_prefers_lower_position():
    """When all positions have identical confidence, the lowest-index position
    in the response window must be committed first.
    """
    vocab_size = 50
    mask_id = vocab_size - 1
    eos_id = vocab_size - 2
    bos_id = vocab_size - 3
    model = _ConstLogitLM(vocab_size).eval()
    tok = _StubTokenizer(vocab_size, mask_id, eos_id, bos_id)
    sampler = DeterministicMDLMSampler(model=model, tokenizer=tok)

    cfg = DeterministicMDLMSamplerConfig(
        teacher_steps=4, max_response_len=4, top_k=4
    )
    prompts = [torch.tensor([7, 8], dtype=torch.long)]
    out = sampler.sample_with_trajectory(prompts, cfg)

    # First step: only the lowest-index masked response position should be
    # committed. mask_state at step 0 has all 4 response positions masked
    # (this is recorded BEFORE commit). At step 1, position 0 (relative
    # to response start) should be the only one not masked -- assuming
    # the schedule commits 1 token per step, which it does for 4 positions
    # / 4 steps.
    step0 = out.trajectories[0][0]
    step1 = out.trajectories[0][1]
    assert step0.mask_state.tolist() == [True, True, True, True]
    # After step 0's commit, position 0 should be the one removed.
    assert step1.mask_state.tolist() == [False, True, True, True], (
        f"expected position 0 committed first; got mask_state={step1.mask_state}"
    )


# ---------------------------------------------------------------------------
# Sampler config hash payload
# ---------------------------------------------------------------------------


def test_sampler_hash_payload_includes_determinism_fields():
    cfg = DeterministicMDLMSamplerConfig(teacher_steps=256, max_response_len=256)
    payload = sampler_config_hash_payload(cfg)
    for must_have in (
        "teacher_steps",
        "max_response_len",
        "top_k",
        "block_size",
        "temperature",
        "stochastic_transfer",
        "remasking",
        "tie_break_method",
    ):
        assert must_have in payload, f"missing {must_have}"


def test_sampler_hash_changes_with_top_k():
    from dllm.pipelines.llada_looped.cache_io import compute_sampler_config_hash

    cfg64 = DeterministicMDLMSamplerConfig(top_k=64)
    cfg128 = DeterministicMDLMSamplerConfig(top_k=128)
    h64 = compute_sampler_config_hash(sampler_config_hash_payload(cfg64))
    h128 = compute_sampler_config_hash(sampler_config_hash_payload(cfg128))
    assert h64 != h128


# ---------------------------------------------------------------------------
# Tie-break: stable lex sort never reorders fp32-distinct values
# ---------------------------------------------------------------------------


def test_select_top_k_does_not_reorder_fp32_distinct_low_confidence():
    """fp32-distinct confidences at low magnitude (e.g. 1e-6 range, where the
    fp32 ULP gap is ~1e-13) must not be reordered by the tie-break path.

    The earlier ``-eps * pos`` shape with eps=1e-12 and T=4096 produced a
    total offset of ~4e-9, which exceeds these gaps and could flip
    adjacent values. The stable-argsort path has no eps so it cannot.
    """
    from dllm.core.samplers.mdlm_deterministic import select_top_k_commit_indices

    T = 4096
    # Build confidence values whose differences (max - second-max etc) are
    # ~1 ULP at magnitude 1e-6 (8e-13). Lay them in the array such that a
    # naive ``-eps * pos`` would flip them.
    base = 1e-6
    eps_fp32 = np.spacing(np.float32(base))  # ~8.5e-14 around 1e-6
    confidence = torch.tensor(
        [base + i * eps_fp32 for i in range(T)],
        dtype=torch.float32,
    ).unsqueeze(0)
    # Reverse: largest at position 0, smallest at position T-1.
    confidence = confidence.flip(dims=[1])
    selectable = torch.ones((1, T), dtype=torch.bool)
    num_unmask = torch.tensor([1], dtype=torch.long)

    transfer = select_top_k_commit_indices(
        confidence_fp32=confidence,
        selectable_mask=selectable,
        num_unmask_per_sample=num_unmask,
    )
    # Position 0 has the largest confidence; it must be the one picked.
    chosen = int(transfer[0].nonzero(as_tuple=False).item())
    assert chosen == 0, (
        f"expected position 0 (highest confidence) to be picked; got {chosen}. "
        "Tie-break implementation is reordering fp32-distinct values."
    )


def test_select_top_k_breaks_exact_ties_by_lower_index():
    """All-equal confidence -> pick lowest-index first."""
    from dllm.core.samplers.mdlm_deterministic import select_top_k_commit_indices

    T = 32
    confidence = torch.full((1, T), 0.5, dtype=torch.float32)
    selectable = torch.ones((1, T), dtype=torch.bool)
    transfer = select_top_k_commit_indices(
        confidence_fp32=confidence,
        selectable_mask=selectable,
        num_unmask_per_sample=torch.tensor([3], dtype=torch.long),
    )
    chosen = transfer[0].nonzero(as_tuple=False).flatten().tolist()
    assert chosen == [0, 1, 2], chosen


def test_select_top_k_skips_non_selectable():
    from dllm.core.samplers.mdlm_deterministic import select_top_k_commit_indices

    confidence = torch.tensor([[0.9, 0.95, 0.7, 0.8, 0.6]], dtype=torch.float32)
    selectable = torch.tensor([[True, False, True, True, True]])
    transfer = select_top_k_commit_indices(
        confidence_fp32=confidence,
        selectable_mask=selectable,
        num_unmask_per_sample=torch.tensor([2], dtype=torch.long),
    )
    chosen = set(transfer[0].nonzero(as_tuple=False).flatten().tolist())
    # Top among selectable: 0 (0.9), 3 (0.8). Position 1 (0.95) is masked out.
    assert chosen == {0, 3}, chosen


# ---------------------------------------------------------------------------
# Trajectory shape sanity
# ---------------------------------------------------------------------------


def test_trajectory_shapes_match_config():
    sampler = _make_sampler()
    cfg = DeterministicMDLMSamplerConfig(
        teacher_steps=6, max_response_len=6, top_k=8
    )
    prompts = [torch.tensor([1, 2, 3], dtype=torch.long)]
    out = sampler.sample_with_trajectory(prompts, cfg)
    step = out.trajectories[0][0]
    assert step.mask_state.shape == (cfg.max_response_len,)
    assert step.top_k_indices.shape == (cfg.max_response_len, cfg.top_k)
    assert step.top_k_logprobs.shape == (cfg.max_response_len, cfg.top_k)
    assert step.neg_log_tail_mass.shape == (cfg.max_response_len,)
    assert step.top1_conf.shape == (cfg.max_response_len,)
    assert step.entropy.shape == (cfg.max_response_len,)
