"""Coarse-KL math tests.

How to run::

    cd /Users/pixeli/dllm
    python -m pytest tests/td/test_coarse_kl.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dllm.core.trainers.td_losses import coarse_kl_with_tail


# ---------------------------------------------------------------------------
# Helpers: build (teacher_top_k_*, neg_log_tail) from a full prob tensor
# ---------------------------------------------------------------------------


def _teacher_targets_from_probs(teacher_probs: torch.Tensor, K: int):
    """Pick top-K and compute the tail bucket exactly. fp32 in, fp32 out."""
    assert teacher_probs.dtype == torch.float32
    top_k_probs, top_k_idx = teacher_probs.topk(K, dim=-1)
    tail_mass = (1.0 - top_k_probs.sum(-1)).clamp(min=1e-30)
    top_k_logprobs = top_k_probs.clamp(min=1e-30).log()
    neg_log_tail = -tail_mass.log()
    return top_k_idx.long(), top_k_logprobs, neg_log_tail


def _full_softmax_kl(student_logits: torch.Tensor, teacher_probs: torch.Tensor):
    """Reference: full-vocab KL(teacher || student) using F.kl_div."""
    student_lp = F.log_softmax(student_logits, dim=-1)
    teacher_lp = teacher_probs.clamp(min=1e-30).log()
    # KL(teacher || student) = sum teacher * (log teacher - log student)
    kl = (teacher_probs * (teacher_lp - student_lp)).sum(-1)
    return kl


# ---------------------------------------------------------------------------
# Equivalence to full-vocab KL when K = V
# ---------------------------------------------------------------------------


def test_coarse_kl_at_K_equal_V_matches_full_softmax_kl():
    """When K == V, the tail bucket is empty and coarse KL = full KL."""
    torch.manual_seed(0)
    N, V = 7, 50
    K = V  # all of the vocabulary

    student_logits = torch.randn(N, V, dtype=torch.float32) * 2.0
    teacher_logits = torch.randn(N, V, dtype=torch.float32) * 2.0
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)

    coarse = coarse_kl_with_tail(
        student_logits, top_k_idx, top_k_lp, neg_log_tail, reduction="none"
    )
    full = _full_softmax_kl(student_logits, teacher_probs)

    # Tail mass is ~1e-30 (clamped), so the tail term contributes near 0.
    torch.testing.assert_close(coarse, full, rtol=1e-4, atol=1e-5)


def test_coarse_kl_close_to_full_kl_when_top_k_covers_most_mass():
    """K << V is the realistic regime; coarse KL upper-bounds full KL well
    when top-K mass is high.
    """
    torch.manual_seed(1)
    N, V, K = 4, 200, 32

    teacher_logits = torch.randn(N, V, dtype=torch.float32) * 5.0  # peaky
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    top_k_mass = teacher_probs.topk(K, dim=-1).values.sum(-1)
    assert top_k_mass.min() > 0.95, "test setup: top-K should cover most mass"

    student_logits = teacher_logits + torch.randn_like(teacher_logits) * 0.3
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)

    coarse = coarse_kl_with_tail(
        student_logits, top_k_idx, top_k_lp, neg_log_tail, reduction="none"
    )
    full = _full_softmax_kl(student_logits, teacher_probs)

    # Both should be small and nearly equal at high top-K mass.
    assert (coarse >= 0).all()
    # Difference per position should be small.
    diff = (coarse - full).abs()
    assert diff.max() < 0.05, f"coarse vs full KL diff too large: {diff.max():.4f}"


# ---------------------------------------------------------------------------
# Zero / positivity
# ---------------------------------------------------------------------------


def test_coarse_kl_zero_when_student_matches_teacher():
    """If student logits equal teacher logits, coarse KL = 0 (modulo eps)."""
    torch.manual_seed(2)
    N, V, K = 3, 100, 32
    teacher_logits = torch.randn(N, V, dtype=torch.float32) * 3.0
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)

    student_logits = teacher_logits.clone()
    kl = coarse_kl_with_tail(
        student_logits, top_k_idx, top_k_lp, neg_log_tail, reduction="none"
    )
    # Student top-K mass under student's own softmax matches teacher's
    # exactly (clamps at 1 - 1e-6 if mass ~ 1). So KL ~= 0.
    assert kl.max().item() < 1e-3


def test_coarse_kl_positive_otherwise():
    torch.manual_seed(3)
    N, V, K = 5, 80, 16
    teacher_logits = torch.randn(N, V, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)

    # Uniform student
    student_logits = torch.zeros_like(teacher_logits)

    kl = coarse_kl_with_tail(
        student_logits, top_k_idx, top_k_lp, neg_log_tail, reduction="none"
    )
    assert (kl >= 0).all()
    assert kl.mean().item() > 1e-2


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------


def test_coarse_kl_handles_perfect_student():
    """Student is one-hot on a token in teacher's top-K -> finite, non-NaN."""
    N, V, K = 2, 50, 8
    teacher_logits = torch.randn(N, V, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)

    # Student spike on teacher's top-1 token at every position.
    student_logits = torch.full((N, V), -50.0, dtype=torch.float32)
    top1_idx = top_k_idx[:, 0]
    for n in range(N):
        student_logits[n, top1_idx[n]] = 50.0

    kl = coarse_kl_with_tail(
        student_logits, top_k_idx, top_k_lp, neg_log_tail, reduction="none"
    )
    assert torch.isfinite(kl).all(), f"NaN/Inf in KL: {kl}"


def test_coarse_kl_gradient_flows_through_student():
    torch.manual_seed(4)
    N, V, K = 3, 60, 16
    teacher_logits = torch.randn(N, V, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)

    student_logits = torch.randn(N, V, dtype=torch.float32, requires_grad=True)
    loss = coarse_kl_with_tail(student_logits, top_k_idx, top_k_lp, neg_log_tail)
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert student_logits.grad.abs().sum() > 0


def test_coarse_kl_rejects_non_fp32_student():
    """Mis-cast at the call site is a hard error (REENTRY_TD_SPEC.md §13)."""
    N, V, K = 2, 30, 8
    teacher_logits = torch.randn(N, V, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)
    student_logits = torch.randn(N, V, dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="fp32"):
        coarse_kl_with_tail(student_logits, top_k_idx, top_k_lp, neg_log_tail)


# ---------------------------------------------------------------------------
# Reductions / shapes
# ---------------------------------------------------------------------------


def test_coarse_kl_reductions():
    torch.manual_seed(5)
    N, V, K = 4, 40, 8
    teacher_logits = torch.randn(N, V, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)
    student_logits = torch.randn(N, V, dtype=torch.float32)

    none_loss = coarse_kl_with_tail(
        student_logits, top_k_idx, top_k_lp, neg_log_tail, reduction="none"
    )
    mean_loss = coarse_kl_with_tail(
        student_logits, top_k_idx, top_k_lp, neg_log_tail, reduction="mean"
    )
    sum_loss = coarse_kl_with_tail(
        student_logits, top_k_idx, top_k_lp, neg_log_tail, reduction="sum"
    )

    assert none_loss.shape == (N,)
    assert mean_loss.shape == ()
    assert sum_loss.shape == ()
    torch.testing.assert_close(none_loss.mean(), mean_loss)
    torch.testing.assert_close(none_loss.sum(), sum_loss)


def test_coarse_kl_unknown_reduction_raises():
    N, V, K = 1, 10, 4
    teacher_probs = F.softmax(torch.randn(N, V), dim=-1)
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, K)
    with pytest.raises(ValueError, match="reduction"):
        coarse_kl_with_tail(
            torch.randn(N, V),
            top_k_idx,
            top_k_lp,
            neg_log_tail,
            reduction="median",
        )


def test_coarse_kl_singleton_batch():
    """Edge case: N = 1 sometimes triggers shape bugs in fancy indexing."""
    teacher_logits = torch.randn(1, 30, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    top_k_idx, top_k_lp, neg_log_tail = _teacher_targets_from_probs(teacher_probs, 5)
    student_logits = torch.randn(1, 30, dtype=torch.float32)
    kl = coarse_kl_with_tail(student_logits, top_k_idx, top_k_lp, neg_log_tail)
    assert torch.isfinite(kl)
