"""Coarse KL with tail bucket -- the only loss P0/P1/P2 need.

How to run the unit tests::

    python -m pytest tests/td/test_coarse_kl.py -v

Math (REENTRY_TD_SPEC.md §6.1):

For each supervised position we have

* teacher's top-K log probabilities ``q_top``  (shape [N, K])
* teacher's top-K token indices       ``idx``  (shape [N, K])
* teacher's tail mass stored as       ``-log(1 - sum q_top)``  (shape [N])

We compute the student's log probability at the same K indices and
re-normalise the student's tail likewise. KL is then summed over K + 1
buckets:

    KL = sum_k q_k * (log q_k - log p_k)  +  q_tail * (log q_tail - log p_tail)

Inputs **must** be cast to fp32 before calling -- bf16 underflows when
``q_k`` is very small. The caller is responsible for that cast.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coarse_kl_with_tail(
    student_logits: torch.Tensor,
    teacher_top_k_indices: torch.Tensor,
    teacher_top_k_logprobs: torch.Tensor,
    teacher_neg_log_tail_mass: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-position coarse KL with one tail bucket.

    All tensors must be on the same device. Numerical fields must be
    fp32 (no internal upcast -- caller's responsibility).

    Args:
        student_logits: ``[N, V]`` -- student's logits at N supervised
            positions. fp32.
        teacher_top_k_indices: ``[N, K]`` int64 -- token ids of teacher's
            top-K at each position.
        teacher_top_k_logprobs: ``[N, K]`` fp32 -- log p_teacher at those
            indices, normalised so ``exp(.).sum(-1) <= 1``. The complement
            is the tail bucket.
        teacher_neg_log_tail_mass: ``[N]`` fp32 -- ``-log(1 - sum top-K)``.
            Use a positive sign so ``log_tail = -neg_log_tail`` and
            ``tail_mass = exp(-neg_log_tail)``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"`` over the N dim.

    Returns:
        Scalar (mean / sum) or ``[N]`` (none). KL contribution per
        position is non-negative.
    """
    if student_logits.dtype != torch.float32:
        raise TypeError(
            "student_logits must be fp32. Cast at the call site to make "
            "the precision boundary explicit."
        )

    # Student log-probs over the full vocab (fp32).
    student_logprobs = F.log_softmax(student_logits, dim=-1)  # [N, V]
    student_top_k_lp = student_logprobs.gather(-1, teacher_top_k_indices)  # [N, K]

    # Student's tail = 1 - exp(top_k_lp).sum(). Clamp so log() is finite
    # at "student is perfectly confident on this top-K".
    student_top_k_mass = student_top_k_lp.exp().sum(dim=-1).clamp(max=1.0 - 1e-6)
    student_log_tail = (1.0 - student_top_k_mass).clamp(min=1e-30).log()  # [N]

    teacher_top_k_p = teacher_top_k_logprobs.exp()  # [N, K]
    teacher_log_tail = -teacher_neg_log_tail_mass  # [N]
    teacher_tail_p = teacher_log_tail.exp()  # [N]

    kl_top = (teacher_top_k_p * (teacher_top_k_logprobs - student_top_k_lp)).sum(-1)  # [N]
    kl_tail = teacher_tail_p * (teacher_log_tail - student_log_tail)  # [N]
    kl = kl_top + kl_tail  # [N]

    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    if reduction == "none":
        return kl
    raise ValueError(f"unknown reduction {reduction!r}")
