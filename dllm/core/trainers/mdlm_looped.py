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

4. Warm-start freeze. When `freeze_base=True`, only the loop-controller
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

        assert self.mu_rec_train_min >= 1
        assert self.mu_rec_train_max >= self.mu_rec_train_min
        assert 0.0 <= self.mu_bwd_ratio <= 1.0
        assert self.recurrence_dist in ("poisson", "uniform", "fixed")

        if bool(args.freeze_base):
            self._apply_base_freeze(True)

        # Running stats reset at each log emit (see `log`).
        self._loop_running = {
            "t_rec_sum": 0.0,
            "t_bwd_sum": 0.0,
            "n": 0,
        }

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
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

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

        # --- Accumulate running stats for `log()` emit ---
        if model.training:
            self._loop_running["t_rec_sum"] += float(T_rec)
            self._loop_running["t_bwd_sum"] += float(T_bwd)
            self._loop_running["n"] += 1

        return (main_loss, outputs) if return_outputs else main_loss

    # ---- Log injection -------------------------------------------------------
    def log(self, logs: dict, start_time=None) -> None:  # type: ignore[override]
        """Inject loop-specific metrics into every trainer log emit."""
        # Running averages over the interval since the previous log.
        n = max(1, self._loop_running["n"])
        logs["loop/T_rec_avg"] = self._loop_running["t_rec_sum"] / n
        logs["loop/T_bwd_avg"] = self._loop_running["t_bwd_sum"] / n
        # Reset running counters.
        for k in self._loop_running:
            self._loop_running[k] = 0 if k == "n" else 0.0

        # Static snapshots of loop params (FSDP-safe, each rank contributes its shard).
        logs.update(self._loop_param_snapshot())

        # Forward to base Trainer.log (handles TB/wandb emit).
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            # Older HF versions (pre-4.46) don't take start_time.
            super().log(logs)

    # ---- Snapshot helpers ----------------------------------------------------
    @staticmethod
    def _reduce_stats(local: torch.Tensor) -> dict:
        """Global mean / std / absmax of `local` (a shard or full tensor).
        Uses all_reduce when torch.distributed is initialized; otherwise uses
        the local values directly."""
        local = local.detach().float()
        n = torch.tensor(float(local.numel()), device=local.device)
        s = local.sum()
        s2 = local.pow(2).sum()
        mx = local.abs().max() if local.numel() > 0 else torch.tensor(
            0.0, device=local.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            pack = torch.stack([s, s2, n])
            torch.distributed.all_reduce(pack, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(mx, op=torch.distributed.ReduceOp.MAX)
            s_, s2_, n_ = pack[0].item(), pack[1].item(), pack[2].item()
            mx_ = mx.item()
        else:
            s_, s2_, n_, mx_ = s.item(), s2.item(), n.item(), mx.item()
        n_ = max(n_, 1.0)
        mean = s_ / n_
        var = max(0.0, s2_ / n_ - mean * mean)
        return {"mean": mean, "std": var ** 0.5, "absmax": mx_}

    def _loop_param_snapshot(self) -> dict:
        """Per-param + derived stats for log_A, delta, B_inject, plus gradient
        norms. Safe under FSDP with fsdp_use_orig_params=True."""
        out: dict = {}
        target_short = {"log_A", "delta", "B_inject"}
        for name, p in self.model.named_parameters():
            short = name.rsplit(".", 1)[-1]
            if short not in target_short:
                continue
            stats = self._reduce_stats(p.data)
            out[f"loop/{short}_mean"] = stats["mean"]
            out[f"loop/{short}_std"] = stats["std"]
            out[f"loop/{short}_absmax"] = stats["absmax"]
            # Gradient norm (if populated).
            if p.grad is not None:
                g = p.grad.detach().float()
                g_sq = g.pow(2).sum()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        g_sq, op=torch.distributed.ReduceOp.SUM
                    )
                out[f"loop/grad_{short}_norm"] = g_sq.item() ** 0.5

        # Cheap scalar proxy for A_disc, computed from the global means of
        # log_A and delta. Not identical to element-wise mean(A_disc), but
        # tracks the same trend and avoids an extra all_gather of the full
        # parameter vector on each log step.
        if "loop/log_A_mean" in out and "loop/delta_mean" in out:
            try:
                a_cont = -math.exp(out["loop/log_A_mean"])
                out["loop/A_disc_proxy"] = math.exp(out["loop/delta_mean"] * a_cont)
            except (OverflowError, ValueError):
                pass

        return out
