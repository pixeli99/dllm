"""
MDLM-Looped trainer.

Extends MDLMTrainer for the belief-bottleneck looped LLaDA:

  1. Per-batch T_rec sampling (uniform over [t_rec_min, t_rec_max]).
     Same T_rec for the full batch (otherwise batched forward is impossible).

  2. 1-step truncated BPTT inside the model. The trainer toggles
     `output_intermediate_logits=True` only when traj_beta > 0 (memory
     cost is O(T_rec * B * L * V)).

  3. Trajectory loss with NORMALIZED weights summing to 1, regardless of
     T_rec, so per-batch loss magnitude does not implicitly depend on
     sampled T_rec:
         w_r = (r/T_rec) / sum_{r'=1..T_rec}(r'/T_rec)   (Schedule A)
         w_r = 1/T_rec                                    (Schedule B)
     Default off (traj_beta=0) -- run a baseline first, then turn on.

  4. Stage 1 / Stage 2 freeze.
     Stage 1: alpha is frozen at 0, T_rec forced to 1. Only LN_out + tau
              (fixed scalar) + lm_head + base finetune. This is the
              probe-train phase that ensures drop-in equivalence.
     Stage 2: alpha unfreezes, T_rec randomized per batch. alpha may be
              linearly warmed up over `alpha_warmup_steps` to reduce
              early instability.

  5. Loop-param LR multiplier (alpha + LN_out + belief_norm).
     10x is conservative; 50x is aggressive.

  6. Diagnostic logging to TensorBoard:
         loop/alpha
         loop/entropy_iter_{r}, loop/kl_iter_{r}, loop/resnorm_iter_{r}
     Use these to diagnose tau collapse, fixed-point convergence, and
     gradient health of the loop.

Run:
    # Imported by the SFT entry:
    #   /Users/pixeli/dllm/examples/llada_looped/sft.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.core.trainers.mdlm import MDLMConfig, MDLMTrainer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MDLMLoopedConfig(MDLMConfig):
    # ---- T_rec sampling at training time ----
    # Uniform over [t_rec_min, t_rec_max] inclusive. Default {1..6} matches
    # the planned grid and lets eval extrapolate to T_rec >= 8.
    t_rec_min: int = 1
    t_rec_max: int = 6
    # Eval-time T_rec passed to model.forward when computing eval loss.
    # The model's config has its own `mu_rec_eval`; this value overrides it
    # if non-zero.
    t_rec_eval: int = 4

    # ---- Trajectory loss ----
    # If > 0, the model is asked to also output intermediate per-iter logits
    # and the trainer adds beta * sum_r w_r * CE(inter_logits_r) to the loss.
    traj_beta: float = 0.0
    # "schedule_a": w_r ~ r / T_rec  (later iters weighted more)
    # "schedule_b": w_r = 1 / T_rec   (uniform)
    # Both normalized so sum_r w_r = 1, regardless of T_rec.
    traj_schedule: str = "schedule_a"

    # ---- Stage 1 / Stage 2 ----
    # When True: freeze alpha (and tau is already a buffer), force T_rec=1.
    # Use for the warm-up phase that probe-trains LN_out + lm_head while
    # preserving exact vanilla-LLaDA equivalence. Then resume with False.
    stage1_freeze_alpha: bool = False
    # When > 0, alpha is linearly ramped from 0 to its current learnable
    # value over the first `alpha_warmup_steps` Stage-2 steps. Provides a
    # gentler entry into the looped regime than a hard unfreeze.
    alpha_warmup_steps: int = 0

    # ---- LR multiplier for loop params ----
    # Loop params are: alpha, ln_out.*, belief_norm.*. tau is a buffer.
    loop_lr_mult: float = 10.0

    # ---- Trainable-surface control ----
    # Freeze prelude / coda transformer blocks. Defaults to True for both:
    #   - prelude is already implicitly frozen by 1-step truncated BPTT
    #     (gradient never reaches blocks[:prelude_layers]); making it
    #     explicit drops 8B-scale optimizer state for those layers.
    #   - coda freeze makes the loop's contribution cleanly attributable
    #     ("R is the only architectural change") and keeps R from chasing
    #     a moving coda target.
    # `freeze_ln_f` and `freeze_wte` left False by default: ln_f is tiny,
    # and wte (= tied lm_head) is needed for both belief readout and
    # final-token decoding -- freezing it would also freeze lm_head.
    freeze_prelude: bool = True
    freeze_coda: bool = True
    freeze_ln_f: bool = False
    freeze_wte: bool = False

    # ---- Diagnostics ----
    # Logging cadence for loop-internal diagnostics (entropy, KL, residual
    # norm). Per-batch logging is overkill; default every 50 steps.
    diag_log_every: int = 50


# Loop-param prefixes (after the HF base_model_prefix="model" wrapping).
_LOOP_PARAM_PREFIXES = (
    "model.alpha",
    "model.ln_out.",
    "model.belief_norm.",
)


def _is_loop_param_name(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in _LOOP_PARAM_PREFIXES)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MDLMLoopedTrainer(MDLMTrainer):

    def __init__(self, args: MDLMLoopedConfig, *pargs, **kwargs):
        super().__init__(args=args, *pargs, **kwargs)

        self.t_rec_min = int(args.t_rec_min)
        self.t_rec_max = int(args.t_rec_max)
        self.t_rec_eval = int(args.t_rec_eval)
        assert self.t_rec_min >= 1
        assert self.t_rec_max >= self.t_rec_min

        self.traj_beta = float(args.traj_beta)
        self.traj_schedule = str(args.traj_schedule)
        assert self.traj_schedule in ("schedule_a", "schedule_b")
        assert self.traj_beta >= 0.0

        self.stage1_freeze_alpha = bool(args.stage1_freeze_alpha)
        self.alpha_warmup_steps = int(args.alpha_warmup_steps)
        self.loop_lr_mult = float(args.loop_lr_mult)
        self.diag_log_every = max(1, int(args.diag_log_every))

        self.freeze_prelude = bool(args.freeze_prelude)
        self.freeze_coda = bool(args.freeze_coda)
        self.freeze_ln_f = bool(args.freeze_ln_f)
        self.freeze_wte = bool(args.freeze_wte)

        # Apply block freeze BEFORE FSDP wraps the model (Trainer.__init__
        # finishes before train() does the wrap, so requires_grad=False set
        # here is honored by the FSDP plan).
        self._apply_block_freeze()

        if self.stage1_freeze_alpha:
            self._freeze_alpha(True)

    # ------------------------------------------------------------------
    # Param freezing
    # ------------------------------------------------------------------

    def _freeze_alpha(self, freeze: bool) -> None:
        for name, p in self.model.named_parameters():
            if name == "model.alpha" or name.endswith(".alpha"):
                p.requires_grad = not freeze

    def _apply_block_freeze(self) -> None:
        """Freeze prelude / coda transformer blocks (and optionally ln_f / wte).

        Prelude is already implicitly frozen by 1-step truncated BPTT, so
        this mainly saves optimizer state. Coda freezing is the substantive
        change: it forces R to produce hidden states that vanilla LLaDA's
        coda already knows how to decode.
        """
        if not (
            self.freeze_prelude or self.freeze_coda
            or self.freeze_ln_f or self.freeze_wte
        ):
            return

        # Walk to the inner LLaDAModel (HF wrapper pattern: LM.model.transformer.blocks).
        inner = self.model
        if hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "config") or not hasattr(inner, "transformer"):
            return  # not the structure we expect; bail silently

        cfg = inner.config
        P = int(getattr(cfg, "prelude_layers", 0))
        R = int(getattr(cfg, "recurrent_layers", 0))
        blocks = inner.transformer.blocks  # ModuleList

        n_frozen_prelude = 0
        n_frozen_coda = 0
        if self.freeze_prelude and P > 0:
            for block in blocks[:P]:
                for p in block.parameters():
                    p.requires_grad = False
                    n_frozen_prelude += p.numel()
        if self.freeze_coda:
            for block in blocks[P + R:]:
                for p in block.parameters():
                    p.requires_grad = False
                    n_frozen_coda += p.numel()

        n_frozen_lnf = 0
        if self.freeze_ln_f and hasattr(inner.transformer, "ln_f"):
            for p in inner.transformer.ln_f.parameters():
                p.requires_grad = False
                n_frozen_lnf += p.numel()

        n_frozen_wte = 0
        if self.freeze_wte and hasattr(inner.transformer, "wte"):
            for p in inner.transformer.wte.parameters():
                p.requires_grad = False
                n_frozen_wte += p.numel()
            if hasattr(inner.transformer, "ff_out"):
                # Untied case: also freeze the lm_head projection.
                for p in inner.transformer.ff_out.parameters():
                    p.requires_grad = False
                    n_frozen_wte += p.numel()

        # Log a one-shot summary.
        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        try:
            from dllm.utils import get_default_logger
            logger = get_default_logger("mdlm_looped")
            logger.info(
                "[freeze] prelude=%d (%d params), coda=%d (%d params), "
                "ln_f=%d, wte=%d. trainable %d / %d (%.2f%%).",
                int(self.freeze_prelude), n_frozen_prelude,
                int(self.freeze_coda), n_frozen_coda,
                n_frozen_lnf, n_frozen_wte,
                n_train, n_total, 100.0 * n_train / max(1, n_total),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Optimizer with loop-param LR multiplier
    # ------------------------------------------------------------------

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Build two param groups: base (default lr) and loop (lr * mult).
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        loop_decay_params: List[nn.Parameter] = []
        loop_no_decay_params: List[nn.Parameter] = []

        # Reuse HF's logic for what gets weight decay.
        decay_param_names = self.get_decay_parameter_names(self.model)

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_loop = _is_loop_param_name(name)
            is_decay = name in decay_param_names
            if is_loop:
                (loop_decay_params if is_decay else loop_no_decay_params).append(p)
            else:
                (decay_params if is_decay else no_decay_params).append(p)

        base_lr = self.args.learning_rate
        loop_lr = base_lr * self.loop_lr_mult
        wd = self.args.weight_decay

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "lr": base_lr, "weight_decay": wd})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0})
        if loop_decay_params:
            param_groups.append({"params": loop_decay_params, "lr": loop_lr, "weight_decay": wd})
        if loop_no_decay_params:
            param_groups.append({"params": loop_no_decay_params, "lr": loop_lr, "weight_decay": 0.0})

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )
        # Don't double-set lr; param_groups override the global lr.
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    # ------------------------------------------------------------------
    # T_rec sampling
    # ------------------------------------------------------------------

    def _sample_t_rec(self, training: bool) -> int:
        if not training:
            return self.t_rec_eval
        if self.stage1_freeze_alpha:
            return 1
        if self.t_rec_min == self.t_rec_max:
            return self.t_rec_min
        return int(torch.randint(self.t_rec_min, self.t_rec_max + 1, ()).item())

    # ------------------------------------------------------------------
    # Trajectory loss weights (sum to 1 regardless of T_rec)
    # ------------------------------------------------------------------

    def _traj_weights(self, T_rec: int, device: torch.device) -> torch.Tensor:
        if self.traj_schedule == "schedule_a":
            # w_r ∝ r, normalized: 2r / (T*(T+1))
            r = torch.arange(1, T_rec + 1, device=device, dtype=torch.float32)
            return r / r.sum()
        elif self.traj_schedule == "schedule_b":
            return torch.full((T_rec,), 1.0 / T_rec, device=device, dtype=torch.float32)
        else:
            raise ValueError(self.traj_schedule)

    # ------------------------------------------------------------------
    # Alpha warmup (Stage 2)
    # ------------------------------------------------------------------

    def _alpha_scale(self) -> float:
        if self.alpha_warmup_steps <= 0:
            return 1.0
        step = max(0, int(self.state.global_step))
        return min(1.0, step / float(self.alpha_warmup_steps))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Mirror MDLMTrainer.compute_loss but pass T_rec / mask_token_id /
        intermediate-logit toggles, and add traj_loss when configured."""

        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100

        # 1. Sample diffusion timesteps
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. Apply stochastic masking
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # 3. Forward pass with looped extras
        T_rec = self._sample_t_rec(training=model.training)
        want_intermediate = (self.traj_beta > 0.0)
        want_diag = (
            getattr(self.state, "global_step", 0) % self.diag_log_every == 0
        ) and model.training
        alpha_scale = self._alpha_scale() if model.training else 1.0

        outputs = model(
            input_ids=noised_input_ids,
            attention_mask=attention_mask,
            T_rec=T_rec,
            mask_token_id=self.processing_class.mask_token_id,
            alpha_scale=alpha_scale,
            output_intermediate_logits=want_intermediate,
            output_loop_diagnostics=want_diag,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # 4. Loss weights
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        # 5. L_final
        token_nll_final = F.cross_entropy(
            logits.transpose(1, 2),
            input_ids,
            reduction="none",
        )
        token_nll_final = (
            token_nll_final * loss_weights * masked_mask.to(token_nll_final.dtype)
        )

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll_final.detach(),
            weight=maskable_mask.to(dtype=logits.dtype).detach(),
        )

        # 6. Normalize L_final
        if self.loss_norm_type == "token":
            denom = maskable_mask.sum().clamp_min(1)
            token_nll_final = token_nll_final / denom
        elif self.loss_norm_type == "sequence":
            token_nll_final = token_nll_final / (
                maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
            )
        elif self.loss_norm_type == "batch":
            token_nll_final = token_nll_final / b
        else:
            raise ValueError("Invalid loss_norm_type.")

        loss_final = token_nll_final.sum()

        # 7. L_traj (optional)
        loss_traj = torch.zeros((), device=loss_final.device, dtype=loss_final.dtype)
        if want_intermediate and outputs.intermediate_logits is not None:
            inter_list = outputs.intermediate_logits  # list of [B, L, V]
            T_actual = len(inter_list)
            if T_actual > 0:
                weights = self._traj_weights(T_actual, device=loss_final.device)
                for r, inter_logits in enumerate(inter_list):
                    token_nll_r = F.cross_entropy(
                        inter_logits.transpose(1, 2),
                        input_ids,
                        reduction="none",
                    )
                    token_nll_r = (
                        token_nll_r * loss_weights * masked_mask.to(token_nll_r.dtype)
                    )
                    if self.loss_norm_type == "token":
                        token_nll_r = token_nll_r / maskable_mask.sum().clamp_min(1)
                    elif self.loss_norm_type == "sequence":
                        token_nll_r = token_nll_r / (
                            maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
                        )
                    elif self.loss_norm_type == "batch":
                        token_nll_r = token_nll_r / b
                    loss_traj = loss_traj + weights[r] * token_nll_r.sum()

        loss = loss_final + self.traj_beta * loss_traj

        # 8. Diagnostic logging
        if want_diag and outputs.loop_diagnostics is not None:
            diag = dict(outputs.loop_diagnostics)
            diag["alpha_scale"] = alpha_scale
            self._log_loop_diagnostics(diag, T_rec=T_rec)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Diagnostic logging
    # ------------------------------------------------------------------

    def _log_loop_diagnostics(self, diag: Dict[str, Any], T_rec: int) -> None:
        # Trainer.log() routes to all configured reporters (TensorBoard, WandB, ...).
        # Keep keys flat (no nesting) for compatibility with HF Trainer's logging.
        log_dict: Dict[str, float] = {}

        alpha = diag.get("alpha")
        if alpha is not None:
            try:
                log_dict["loop/alpha"] = float(alpha.item())
            except Exception:
                pass

        tau = diag.get("tau")
        if tau is not None:
            try:
                log_dict["loop/tau"] = float(tau.item())
            except Exception:
                pass

        log_dict["loop/T_rec"] = float(T_rec)

        alpha_scale = diag.get("alpha_scale")
        if alpha_scale is not None:
            log_dict["loop/alpha_scale"] = float(alpha_scale)

        for key in ("entropy", "kl_successive", "residual_norm"):
            seq = diag.get(key, None)
            if not seq:
                continue
            for i, v in enumerate(seq):
                try:
                    log_dict[f"loop/{key}_iter{i}"] = float(v.item())
                except Exception:
                    pass

        if log_dict:
            self.log(log_dict)
