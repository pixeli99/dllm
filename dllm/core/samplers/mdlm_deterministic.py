"""Deterministic MDLM sampler with full per-step trajectory recording.

How to run a quick demo (needs a tokenizer + model on disk)::

    from dllm.core.samplers.mdlm_deterministic import (
        DeterministicMDLMSampler, DeterministicMDLMSamplerConfig,
    )
    sampler = DeterministicMDLMSampler(model=teacher, tokenizer=tok)
    out = sampler.sample_with_trajectory(
        prompts=[tok("1+1=", return_tensors="pt").input_ids[0]],
        config=DeterministicMDLMSamplerConfig(teacher_steps=8, max_response_len=8, top_k=4),
    )

This is intentionally a stand-alone sampler, not a subclass of
:class:`dllm.core.samplers.mdlm.MDLMSampler`. The default sampler does
its commit selection in the model's native dtype (typically bf16) and
has no hook for "record teacher's belief before commit" -- so we'd be
overriding most of its body anyway. The two modules share helpers
(:func:`record_per_position_distribution` and
:func:`select_top_k_commit_indices` are public so MDLMSampler can adopt
them later if a deterministic mode there is wanted), but the sampler
classes themselves are independent. Keep them in sync if you change
the commit math.

Determinism rules (REENTRY_TD_SPEC.md §5.1):

1. softmax + max for confidence is run in fp32 so the per-position
   ranking does not drift with the model's compute dtype.
2. Top-K commit selection is a true lexicographic sort: stable
   descending argsort on fp32 confidence, with ties broken by position
   index ascending. No epsilon arithmetic, so fp32-distinct confidences
   are never reordered. Earlier drafts used ``-eps * pos_idx`` which
   could flip adjacent fp32 values when their gap dropped near
   ``vocab_size**-1`` (typical at low-confidence positions).
3. Temperature = 0 and stochastic_transfer = False are hard-coded; no
   flags. Any randomness here would defeat the cache's reproducibility
   contract.

The model forward itself stays in its native dtype (bf16); only the
softmax + topk path is promoted.

Recorded outputs are serialized via
:mod:`dllm.pipelines.llada_looped.cache_io`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import get_num_transfer_tokens


# ---------------------------------------------------------------------------
# Config + output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeterministicMDLMSamplerConfig(BaseSamplerConfig):
    """Sampler config for trajectory cache builds."""

    teacher_steps: int = 256
    max_response_len: int = 256
    top_k: int = 64
    # Default: 1 block covering the whole response. The split-block path
    # (block_size < max_response_len) still works but is not exercised
    # by the cache builder today.
    block_size: int | None = None
    record_trajectory: bool = True

    # Hardcoded constants surfaced here so the sampler-config hash sees
    # them. If any of these change, cached trajectories are invalidated.
    _temperature: float = 0.0
    _stochastic_transfer: bool = False
    _remasking: str = "low_confidence"
    # Tie-break is true lexicographic (stable argsort), no epsilon. This
    # field is part of the hash payload so a future move away from stable
    # argsort would correctly invalidate caches.
    _tie_break_method: str = "stable_lex"


@dataclass
class TrajectoryStep:
    """A single step's record for ONE sample, response positions only."""

    step_idx: int
    mask_state: np.ndarray  # bool [L_resp]
    top_k_indices: np.ndarray  # int32 [L_resp, K]
    top_k_logprobs: np.ndarray  # fp32 [L_resp, K]
    neg_log_tail_mass: np.ndarray  # fp32 [L_resp]
    top1_conf: np.ndarray  # fp32 [L_resp]
    entropy: np.ndarray  # fp32 [L_resp]


@dataclass
class TrajectorySamplerOutput(BaseSamplerOutput):
    prompt_lens: list[int] = field(default_factory=list)
    trajectories: list[list[TrajectoryStep]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level helpers (shared with future deterministic-mode hooks
# inside MDLMSampler if we ever add one)
# ---------------------------------------------------------------------------


def record_per_position_distribution(
    response_logits: torch.Tensor,
    mask_state: torch.Tensor,
    top_k: int,
) -> dict[str, np.ndarray]:
    """Compute per-position teacher belief stats in fp32, return CPU numpy.

    Args:
        response_logits: ``[L_resp, V]``. Any model dtype; we cast to fp32.
        mask_state: ``[L_resp]`` bool. Currently-masked response positions.
        top_k: how many top tokens to record.

    Returns a dict with the same fields a :class:`TrajectoryStep` carries
    (minus ``step_idx``). Tail mass is stored as ``-log(1 - sum top_k)``,
    floored to avoid ``log(0)`` at very confident positions.
    """
    response_logits = response_logits.float()  # [L_resp, V]
    probs = response_logits.softmax(dim=-1)  # [L_resp, V] fp32
    top_k_probs, top_k_indices = probs.topk(top_k, dim=-1)
    top_k_sum = top_k_probs.sum(-1)

    tail_mass = (1.0 - top_k_sum).clamp(min=1e-30)
    neg_log_tail = -tail_mass.log()
    top_k_logprobs = top_k_probs.clamp(min=1e-30).log()
    top1_conf = top_k_probs[:, 0]
    entropy = -(probs * probs.clamp(min=1e-30).log()).sum(-1)

    return {
        "mask_state": mask_state.detach().cpu().numpy().astype(bool),
        "top_k_indices": top_k_indices.detach().cpu().numpy().astype(np.int32),
        "top_k_logprobs": top_k_logprobs.detach().cpu().numpy().astype(np.float32),
        "neg_log_tail_mass": neg_log_tail.detach().cpu().numpy().astype(np.float32),
        "top1_conf": top1_conf.detach().cpu().numpy().astype(np.float32),
        "entropy": entropy.detach().cpu().numpy().astype(np.float32),
    }


def select_top_k_commit_indices(
    confidence_fp32: torch.Tensor,
    selectable_mask: torch.Tensor,
    num_unmask_per_sample: torch.Tensor,
) -> torch.Tensor:
    """Pick which positions to commit at this step, deterministically.

    Implementation is a true lexicographic sort:

    1. Sort by confidence, descending.
    2. Tie-break by position index, ascending.

    Achieved via a stable descending argsort of fp32 confidence: equal
    values keep their input order, so the lower index naturally wins.
    No epsilon arithmetic, so fp32-distinct confidences cannot be
    reordered -- in particular this is robust at low-confidence
    positions where an earlier ``-eps * pos`` shape could still flip
    adjacent fp32 values when their gap drops to ``~vocab_size**-1``.

    Args:
        confidence_fp32: ``[B, T]`` fp32 -- the max softmax prob per position.
        selectable_mask: ``[B, T]`` bool -- True where the position is
            both currently masked AND inside the active block window.
        num_unmask_per_sample: ``[B]`` int -- how many positions to commit
            for this sample at this step.

    Returns:
        ``[B, T]`` bool -- True at the up-to-num_unmask highest-confidence
        selectable positions per sample.
    """
    if confidence_fp32.dtype != torch.float32:
        raise TypeError(
            "confidence must be fp32 at the helper boundary; the sampler "
            "casts model logits up before computing softmax."
        )

    B, T = confidence_fp32.shape
    device = confidence_fp32.device

    # Mask out non-selectable to -inf so they sort to the end of a
    # *descending* order.
    masked = torch.where(
        selectable_mask, confidence_fp32, torch.full_like(confidence_fp32, -float("inf"))
    )
    # Stable argsort. Ascending sort of -conf == descending stable sort of conf;
    # ties keep input order, so the lower index wins.
    sorted_idx = (-masked).argsort(dim=-1, stable=True)  # [B, T]

    transfer = torch.zeros((B, T), dtype=torch.bool, device=device)
    for j in range(B):
        k = int(num_unmask_per_sample[j].item())
        if k > 0:
            transfer[j, sorted_idx[j, :k]] = True
    return transfer


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


@dataclass
class DeterministicMDLMSampler(BaseSampler):
    """Confidence-based MDLM sampler with deterministic ties + trajectory."""

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        prompts: Sequence[torch.Tensor | list],
        config: DeterministicMDLMSamplerConfig,
    ) -> TrajectorySamplerOutput:
        cfg = config
        block_size = cfg.block_size if cfg.block_size is not None else cfg.max_response_len
        assert 1 <= block_size <= cfg.max_response_len
        assert 1 <= cfg.teacher_steps
        assert 1 <= cfg.top_k

        device = self.model.device
        mask_id = int(self.tokenizer.mask_token_id)
        eos_id = int(self.tokenizer.eos_token_id)

        # ---- Convert prompts to tensors ----
        tensor_prompts: list[torch.Tensor] = []
        for p in prompts:
            if isinstance(p, list):
                p = torch.as_tensor(p, dtype=torch.long, device=device)
            else:
                p = p.to(device=device, dtype=torch.long)
            tensor_prompts.append(p)
        prompt_lens = [int(p.shape[0]) for p in tensor_prompts]
        B = len(tensor_prompts)
        T = max(prompt_lens) + cfg.max_response_len

        # ---- Build canvas: [prompt | mask*L_resp | EOS pad to T] ----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=device)
        for i, p in enumerate(tensor_prompts):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + cfg.max_response_len] = mask_id

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + cfg.max_response_len, T)
            attention_mask[i, :valid_end] = 1

        trajectories: list[list[TrajectoryStep]] = [[] for _ in range(B)]

        num_blocks = math.ceil(cfg.max_response_len / block_size)
        steps_per_block = math.ceil(cfg.teacher_steps / num_blocks)

        global_step_idx = 0
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        pos_t = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

        for b_idx in range(num_blocks):
            block_start_per = prompt_lens_t + b_idx * block_size
            block_end_per = torch.minimum(
                prompt_lens_t + (b_idx + 1) * block_size,
                prompt_lens_t + cfg.max_response_len,
            )

            # Per-sample block-mask map (positions within the block that
            # are still mask tokens RIGHT NOW, before this block starts).
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=device
            )
            for j in range(B):
                start = int(block_start_per[j].item())
                end = int(block_end_per[j].item())
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=False,
            )
            effective_steps = num_transfer_tokens.size(1)

            for step_in_block in range(effective_steps):
                # ===== Forward pass (model dtype) =====
                logits = self.model(x, attention_mask=attention_mask).logits

                # ===== Record trajectory (fp32) =====
                if cfg.record_trajectory:
                    for s in range(B):
                        pl = prompt_lens[s]
                        response_logits = logits[s, pl : pl + cfg.max_response_len]
                        mask_state = x[s, pl : pl + cfg.max_response_len] == mask_id
                        rec = record_per_position_distribution(
                            response_logits=response_logits,
                            mask_state=mask_state,
                            top_k=cfg.top_k,
                        )
                        trajectories[s].append(
                            TrajectoryStep(step_idx=global_step_idx, **rec)
                        )

                # ===== Compute commit (fp32 confidence + stable lex sort) =====
                # argmax in model dtype is fine (logit ties are rare); the
                # selection-vs-confidence ordering is what matters.
                x0 = logits.argmax(dim=-1)
                probs_fp32 = logits.float().softmax(dim=-1)
                conf_fp32 = probs_fp32.max(dim=-1).values  # [B, T]

                mask_index_full = x == mask_id
                in_block = (pos_t >= block_start_per.unsqueeze(1)) & (
                    pos_t < block_end_per.unsqueeze(1)
                )
                selectable = mask_index_full & in_block

                transfer_index = select_top_k_commit_indices(
                    confidence_fp32=conf_fp32,
                    selectable_mask=selectable,
                    num_unmask_per_sample=num_transfer_tokens[:, step_in_block],
                )

                x0 = torch.where(mask_index_full, x0, x)
                x[transfer_index] = x0[transfer_index]

                global_step_idx += 1

        # ---- Sanity: response should be fully unmasked at the end ----
        for j in range(B):
            response = x[j, prompt_lens[j] : prompt_lens[j] + cfg.max_response_len]
            remaining = (response == mask_id).sum().item()
            if remaining != 0:
                raise RuntimeError(
                    f"sample {j}: {remaining} response position(s) still masked "
                    f"after {global_step_idx} steps. Schedule produced too few "
                    "effective commit steps; try increasing teacher_steps or "
                    "reducing max_response_len."
                )

        return TrajectorySamplerOutput(
            sequences=x,
            histories=None,
            prompt_lens=prompt_lens,
            trajectories=trajectories,
        )

    # ------------------------------------------------------------------
    # BaseSampler abstract method satisfaction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        prompts: Sequence[torch.Tensor | list],
        config: DeterministicMDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        if config is None:
            config = DeterministicMDLMSamplerConfig()
        config.record_trajectory = False
        out = self.sample_with_trajectory(prompts, config)
        return BaseSamplerOutput(sequences=out.sequences, histories=None)

    @torch.no_grad()
    def infill(self, *args, **kwargs):
        raise NotImplementedError(
            "DeterministicMDLMSampler is for prompt-conditioned generation only; "
            "infill is not part of the trajectory cache pipeline."
        )


# ---------------------------------------------------------------------------
# Helper: sampler-config payload for hashing into the cache manifest
# ---------------------------------------------------------------------------


def sampler_config_hash_payload(cfg: DeterministicMDLMSamplerConfig) -> dict:
    """Pull out the determinism-relevant fields for hashing.

    Anything that could change a commit decision belongs here.
    """
    return {
        "teacher_steps": cfg.teacher_steps,
        "max_response_len": cfg.max_response_len,
        "top_k": cfg.top_k,
        "block_size": cfg.block_size if cfg.block_size is not None else cfg.max_response_len,
        "temperature": cfg._temperature,
        "stochastic_transfer": cfg._stochastic_transfer,
        "remasking": cfg._remasking,
        "tie_break_method": cfg._tie_break_method,
    }
