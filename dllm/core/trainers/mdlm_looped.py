"""
MDLM-Looped trainer.

Extends MDLMTrainer with:

1. Per-batch recurrence sampling. Each training step samples a number of
   middle-loop iterations `T_rec` from a chosen distribution (Poisson by
   default) and passes it into `model.forward(..., T_rec=T_rec, T_bwd=T_bwd)`.

2. Optional denoising-adaptive schedule `μ_rec(t)`. When
   `t_adaptive=True`, the Poisson mean is modulated by a Gaussian bump
   centred at `t_bump_center` with width `t_bump_width` and peak
   multiplier `t_bump_scale`, so the model spends more layer-loop
   compute at the harder denoising timesteps.

3. Truncated BPTT. Only the last `⌈mu_bwd_ratio * T_rec⌉` recurrent
   iterations propagate gradient; earlier ones run under `torch.no_grad()`
   inside the model.

4. Cycle-consistency regularizer. When
   `cycle_consistency_weight > 0`, a second forward at `T_rec + 1`
   (under `no_grad`) produces a target the main forward is pulled
   toward. This encourages the loop to converge to a fixed point so
   test-time `T_rec` can extrapolate beyond training.

5. Warm-start freeze. When `freeze_base=True`, only the loop-controller
   parameters (`log_A`, `delta`, `B_inject`, `input_norm.*`) are
   trainable. Intended for stage-1 runs before a full unfreeze resume.

Run:
    # This module is imported by the SFT entry:
    #   /Users/pixeli/dllm/examples/llada_looped/sft.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.core.trainers.mdlm import MDLMConfig, MDLMTrainer


@dataclass
class MDLMLoopedConfig(MDLMConfig):
    # ---- Recurrence sampling ----
    mu_rec_train_mean: float = 2.0
    mu_rec_train_min: int = 1
    mu_rec_train_max: int = 4
    recurrence_dist: str = "poisson"  # "poisson" | "uniform" | "fixed"

    # ---- Denoising-adaptive μ_rec(t) ----
    t_adaptive: bool = False
    t_bump_center: float = 0.5
    t_bump_width: float = 0.25
    t_bump_scale: float = 1.0

    # ---- Truncated BPTT ----
    mu_bwd_ratio: float = 0.5

    # ---- Cycle-consistency reg ----
    cycle_consistency_weight: float = 0.0

    # ---- Warm-start freeze ----
    freeze_base: bool = False


_LOOP_PARAM_PREFIXES = (
    "model.log_A",
    "model.delta",
    "model.B_inject",
    "model.input_norm.",
)


def _is_loop_param_name(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in _LOOP_PARAM_PREFIXES)


# ----- Loss-path diagnostic (matches modeling_llada_looped._diag) -----
import os as _os

_TRAINER_DIAG_MAX = int(_os.environ.get("DLLM_LOOP_DIAG_STEPS", "3"))
_TRAINER_DIAG_COUNTER = {"n": 0}


def _diag_loss(t: torch.Tensor, tag: str) -> None:
    has_nan = bool(torch.isnan(t).any().item())
    has_inf = bool(torch.isinf(t).any().item())
    if has_nan or has_inf:
        raise RuntimeError(
            f"[loss-diag] NON-FINITE at '{tag}': "
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"nan={int(torch.isnan(t).sum().item())} "
            f"inf={int(torch.isinf(t).sum().item())}"
        )
    if _TRAINER_DIAG_COUNTER["n"] >= _TRAINER_DIAG_MAX:
        return
    if _os.environ.get("LOCAL_RANK", "0") != "0":
        return
    tf = t.detach().float()
    print(
        f"[loss-diag step={_TRAINER_DIAG_COUNTER['n']}] {tag}: "
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={tf.min().item():+.3e} "
        f"max={tf.max().item():+.3e} "
        f"mean={tf.mean().item():+.3e}",
        flush=True,
    )


class MDLMLoopedTrainer(MDLMTrainer):

    def __init__(self, args: MDLMLoopedConfig, *pargs, **kwargs):
        super().__init__(args=args, *pargs, **kwargs)
        self.mu_rec_train_mean = float(args.mu_rec_train_mean)
        self.mu_rec_train_min = int(args.mu_rec_train_min)
        self.mu_rec_train_max = int(args.mu_rec_train_max)
        self.recurrence_dist = args.recurrence_dist
        self.t_adaptive = bool(args.t_adaptive)
        self.t_bump_center = float(args.t_bump_center)
        self.t_bump_width = float(args.t_bump_width)
        self.t_bump_scale = float(args.t_bump_scale)
        self.mu_bwd_ratio = float(args.mu_bwd_ratio)
        self.cycle_consistency_weight = float(args.cycle_consistency_weight)

        assert self.mu_rec_train_min >= 1
        assert self.mu_rec_train_max >= self.mu_rec_train_min
        assert 0.0 <= self.mu_bwd_ratio <= 1.0
        assert self.recurrence_dist in ("poisson", "uniform", "fixed")

        if bool(args.freeze_base):
            self._apply_base_freeze(True)

    # ---- Freeze/unfreeze base model ----
    def _apply_base_freeze(self, freeze: bool) -> None:
        """Freeze all params except the loop controller when `freeze=True`."""
        for name, p in self.model.named_parameters():
            if freeze:
                p.requires_grad = _is_loop_param_name(name)
            else:
                p.requires_grad = True

    # ---- Sampling μ_rec ----
    def _mu_rec_mean_for_t(self, t_batch: torch.Tensor) -> float:
        if not self.t_adaptive:
            return self.mu_rec_train_mean
        t_mean = float(t_batch.mean().item())
        bump = math.exp(
            -(((t_mean - self.t_bump_center) / max(self.t_bump_width, 1e-6)) ** 2)
        )
        return self.mu_rec_train_mean * (1.0 + self.t_bump_scale * bump)

    def _sample_mu_rec(self, t_batch: torch.Tensor) -> int:
        mean = self._mu_rec_mean_for_t(t_batch)
        if self.recurrence_dist == "poisson":
            t_rec = int(np.random.poisson(lam=max(mean, 0.0)))
        elif self.recurrence_dist == "uniform":
            t_rec = int(
                np.random.randint(self.mu_rec_train_min, self.mu_rec_train_max + 1)
            )
        elif self.recurrence_dist == "fixed":
            t_rec = int(round(mean))
        else:
            raise ValueError(f"Unknown recurrence_dist: {self.recurrence_dist}")
        return max(self.mu_rec_train_min, min(self.mu_rec_train_max, t_rec))

    def _compute_t_bwd(self, t_rec: int) -> int:
        return max(1, int(math.ceil(self.mu_bwd_ratio * t_rec)))

    # ---- Loss ----
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        b, l = input_ids.shape
        maskable_mask = labels != -100

        # --- Sample diffusion timestep and mask ---
        t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # --- Sample μ_rec for this batch, derive truncated-BPTT depth ---
        # During eval use a fixed T_rec so metrics are comparable epoch-to-epoch.
        if model.training:
            T_rec = self._sample_mu_rec(t)
            T_bwd = self._compute_t_bwd(T_rec)
        else:
            T_rec = max(
                self.mu_rec_train_min,
                min(self.mu_rec_train_max, int(round(self.mu_rec_train_mean))),
            )
            T_bwd = 0  # no grad in eval anyway

        # --- Main forward ---
        outputs = model(
            input_ids=noised_input_ids,
            attention_mask=attention_mask,
            T_rec=T_rec,
            T_bwd=T_bwd,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits
        _diag_loss(logits, "A_logits_from_model")

        # Step-level summary (rank 0, first few steps only).
        if (
            _TRAINER_DIAG_COUNTER["n"] < _TRAINER_DIAG_MAX
            and _os.environ.get("LOCAL_RANK", "0") == "0"
        ):
            n_mask = int(masked_mask.sum().item())
            n_maskable = int(maskable_mask.sum().item())
            print(
                f"[loss-diag step={_TRAINER_DIAG_COUNTER['n']}] "
                f"T_rec={T_rec} T_bwd={T_bwd} "
                f"p_mask.mean={p_mask.mean().item():.3f} "
                f"masked_tokens={n_mask} maskable_tokens={n_maskable} "
                f"t.mean={t.mean().item():.3f}",
                flush=True,
            )

        # --- MDLM token-level CE on masked positions ---
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )
        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            logits.transpose(1, 2), input_ids, reduction="none"
        )
        _diag_loss(token_nll, "B_token_nll_raw")
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)
        _diag_loss(token_nll, "C_token_nll_masked")

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=logits.dtype).detach(),
        )

        if self.loss_norm_type == "token":
            token_nll = token_nll / maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll = token_nll / (
                maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
            )
        elif self.loss_norm_type == "batch":
            token_nll = token_nll / b
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.loss_norm_type}")
        main_loss = token_nll.sum()
        _diag_loss(main_loss, "D_main_loss")
        _TRAINER_DIAG_COUNTER["n"] += 1

        total_loss = main_loss

        # --- Cycle-consistency: pull logits(T_rec) toward detached logits(T_rec+1) ---
        if self.cycle_consistency_weight > 0.0 and T_rec < self.mu_rec_train_max:
            with torch.no_grad():
                ref_outputs = model(
                    input_ids=noised_input_ids,
                    attention_mask=attention_mask,
                    T_rec=T_rec + 1,
                    T_bwd=0,
                )
            ref_logits = ref_outputs.logits.detach()
            # Promote to float32 for the diff^2 sum — logits are [B, L, ~126k]
            # bf16, and a bf16 squared diff can overflow / underflow easily.
            mm = masked_mask.to(dtype=torch.float32).unsqueeze(-1)
            denom = mm.sum().clamp_min(1.0)
            diff = logits.float() - ref_logits.float()
            cycle_loss = (diff.pow(2) * mm).sum() / denom
            total_loss = total_loss + self.cycle_consistency_weight * cycle_loss

        return (total_loss, outputs) if return_outputs else total_loss
