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

Why this exists separately from :mod:`dllm.core.samplers.mdlm`:

* The default :class:`MDLMSampler` runs confidence selection in the
  model's dtype (typically bf16) and applies no tie-break, so two runs
  on the same hardware produce different commits whenever ties happen
  (very common at low entropy).
* It also has no hook for "record what teacher believed before this
  commit" -- which is exactly what trajectory distillation needs.

This sampler addresses both:

* fp32 softmax / fp32 confidence throughout the commit-selection path;
* a per-position ``-eps * pos_idx`` tie-break that makes lower indices
  win deterministically (matches REENTRY_TD_SPEC.md §5.1);
* per-(sample, step) recording of top-K logprobs, tail mass,
  argmax confidence, and entropy at response positions only.

The model forward itself stays in its native dtype (bf16). Only the
softmax + topk + tie-break path is forced to fp32.

Output shapes are committed at write time -- see
:mod:`dllm.pipelines.llada_looped.cache_io` for how the trajectories
get serialized into shards.
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
    """Sampler config for trajectory cache builds.

    Determinism requires temperature=0 and stochastic_transfer=False;
    both are hard-coded inside the sampler -- no flags. ``tie_break_eps``
    is exposed for testing the tie-break behaviour.
    """

    teacher_steps: int = 256
    max_response_len: int = 256
    top_k: int = 64
    # Default: 1 block covering the whole response. The split-block path
    # (block_size < max_response_len) still works but is currently not
    # exercised by the cache builder.
    block_size: int | None = None
    record_trajectory: bool = True
    tie_break_eps: float = 1e-7

    # Hardcoded constants we want to surface for hashing the sampler
    # config -- if any of these change, the cache must be rebuilt.
    _temperature: float = 0.0
    _stochastic_transfer: bool = False
    _remasking: str = "low_confidence"


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
    """Sampler output extended with the per-sample trajectory."""

    prompt_lens: list[int] = field(default_factory=list)
    trajectories: list[list[TrajectoryStep]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


@dataclass
class DeterministicMDLMSampler(BaseSampler):
    """Confidence-based MDLM sampler with deterministic ties + trajectory.

    The model forward stays in its native dtype. Everything that affects
    the commit decision (softmax, max, topk, tie-break) runs in fp32.
    """

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        prompts: Sequence[torch.Tensor | list],
        config: DeterministicMDLMSamplerConfig,
    ) -> TrajectorySamplerOutput:
        cfg = config
        if cfg.block_size is None:
            block_size = cfg.max_response_len
        else:
            block_size = cfg.block_size
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

        # ---- Trajectory storage ----
        trajectories: list[list[TrajectoryStep]] = [[] for _ in range(B)]

        # ---- Block scheduling ----
        num_blocks = math.ceil(cfg.max_response_len / block_size)
        steps_per_block = math.ceil(cfg.teacher_steps / num_blocks)

        global_step_idx = 0

        for b_idx in range(num_blocks):
            # Per-sample block-mask map.
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=device
            )
            for j in range(B):
                start = prompt_lens[j] + b_idx * block_size
                end = min(start + block_size, prompt_lens[j] + cfg.max_response_len, T)
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
                logits = self.model(x, attention_mask=attention_mask).logits  # [B, T, V]

                # ===== Record trajectory (fp32) =====
                if cfg.record_trajectory:
                    self._record_step(
                        logits=logits,
                        x=x,
                        prompt_lens=prompt_lens,
                        max_response_len=cfg.max_response_len,
                        top_k=cfg.top_k,
                        mask_id=mask_id,
                        global_step_idx=global_step_idx,
                        trajectories=trajectories,
                    )

                # ===== Compute commit decision (fp32) =====
                # argmax can use logits in their native dtype -- ties at
                # logit level are rare. Confidence (fp32) is what we use
                # to pick which positions to commit.
                x0 = logits.argmax(dim=-1)  # [B, T]

                probs_fp32 = logits.float().softmax(dim=-1)  # [B, T, V] fp32
                conf_full = probs_fp32.max(dim=-1).values  # [B, T] fp32

                mask_index_full = x == mask_id
                neg_inf = torch.full_like(conf_full, -float("inf"))
                confidence = torch.where(mask_index_full, conf_full, neg_inf)

                # Restrict to current block window per sample
                for j in range(B):
                    block_end = prompt_lens[j] + (b_idx + 1) * block_size
                    confidence[j, : prompt_lens[j]] = -float("inf")
                    confidence[j, block_end:] = -float("inf")

                # Tie-break: prefer lower position index
                tie_break = -cfg.tie_break_eps * torch.arange(
                    T, device=device, dtype=torch.float32
                )
                confidence = confidence + tie_break.unsqueeze(0)

                # Per-sample top-K commit
                transfer_index = torch.zeros_like(x, dtype=torch.bool, device=device)
                for j in range(B):
                    k = int(num_transfer_tokens[j, step_in_block].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

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
    # Trajectory recording (broken out for clarity / testability)
    # ------------------------------------------------------------------

    def _record_step(
        self,
        logits: torch.Tensor,  # [B, T, V] in model dtype
        x: torch.Tensor,  # [B, T]
        prompt_lens: list[int],
        max_response_len: int,
        top_k: int,
        mask_id: int,
        global_step_idx: int,
        trajectories: list[list[TrajectoryStep]],
    ) -> None:
        """Compute response-window stats in fp32 and append to ``trajectories``."""
        B = logits.shape[0]

        # Slice each sample's response window. Shapes differ if prompts
        # have different lengths (response_window is L_resp wide for all,
        # just at different offsets), so we loop here rather than gather.
        for s in range(B):
            pl = prompt_lens[s]
            response_logits = logits[s, pl : pl + max_response_len].float()  # [L_resp, V]

            probs = response_logits.softmax(dim=-1)  # [L_resp, V] fp32
            top_k_probs, top_k_indices = probs.topk(top_k, dim=-1)  # [L_resp, K]
            top_k_sum = top_k_probs.sum(-1)  # [L_resp]

            # tail_mass = 1 - sum(top_K). Floor to 1e-30 so log doesn't blow up
            # at very confident positions; downstream KL will see neg_log_tail
            # near log(1e30) ~ 69, well within bf16 range.
            tail_mass = (1.0 - top_k_sum).clamp(min=1e-30)
            neg_log_tail = -tail_mass.log()  # [L_resp]

            top_k_logprobs = top_k_probs.clamp(min=1e-30).log()  # [L_resp, K]
            top1_conf = top_k_probs[:, 0]  # [L_resp]
            entropy = -(probs * probs.clamp(min=1e-30).log()).sum(-1)  # [L_resp]

            mask_state = x[s, pl : pl + max_response_len] == mask_id  # [L_resp] bool

            trajectories[s].append(
                TrajectoryStep(
                    step_idx=global_step_idx,
                    mask_state=mask_state.cpu().numpy().astype(bool),
                    top_k_indices=top_k_indices.cpu().numpy().astype(np.int32),
                    top_k_logprobs=top_k_logprobs.cpu().numpy().astype(np.float32),
                    neg_log_tail_mass=neg_log_tail.cpu().numpy().astype(np.float32),
                    top1_conf=top1_conf.cpu().numpy().astype(np.float32),
                    entropy=entropy.cpu().numpy().astype(np.float32),
                )
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
        """Convenience: run with trajectory recording off, return only sequences."""
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

    Anything that could change a commit decision belongs here. Anything
    that's purely metadata (like ``record_trajectory``) does not.
    """
    return {
        "teacher_steps": cfg.teacher_steps,
        "max_response_len": cfg.max_response_len,
        "top_k": cfg.top_k,
        "block_size": cfg.block_size if cfg.block_size is not None else cfg.max_response_len,
        "tie_break_eps": cfg.tie_break_eps,
        "temperature": cfg._temperature,
        "stochastic_transfer": cfg._stochastic_transfer,
        "remasking": cfg._remasking,
    }
