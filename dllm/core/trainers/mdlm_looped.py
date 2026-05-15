"""
MDLM-Looped trainer.

Extends MDLMTrainer for the latent-feedback looped LLaDA:

  1. Per-batch T_rec sampling (uniform over [t_rec_min, t_rec_max]).
     Same T_rec for the full batch (otherwise batched forward is impossible).

  2. 1-step truncated BPTT inside the model.

  3. Loop-param LR multiplier (recursive_link.* plus optional damping alpha).
     10x is conservative; 50x is aggressive.

  4. Trainable-surface control. Default: freeze prelude + coda transformer
     blocks. This makes the loop's contribution cleanly attributable
     ("R is the only architectural change") and reduces optimizer state.

  5. Diagnostic logging to TensorBoard:
         loop/T_rec
         loop/residual_norm_iter{r}    -- ||h_r - h_{r-1}|| / ||h_{r-1}||
         loop/raw_residual_norm_iter{r} -- undamped proposal movement, if enabled
         loop/adapter_update_norm_iter{r}
         loop/damping_alpha_iter{r}
     Use these to diagnose fixed-point convergence and adapter activity.

Run:
    # Imported by the SFT entry:
    #   /Users/pixeli/dllm/examples/llada_looped/sft.py
"""

from __future__ import annotations

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
    t_rec_min: int = 1
    t_rec_max: int = 6
    # Eval-time T_rec passed to model.forward when computing eval loss.
    t_rec_eval: int = 4

    # ---- LR multiplier for loop params ----
    # Loop params: recursive_link.* plus optional damping_alpha_logit.
    loop_lr_mult: float = 10.0

    # ---- Trainable-surface control ----
    # The default profile (all True) trains ONLY recursive_link plus optional
    # damping_alpha_logit, freezing the entire vanilla-LLaDA backbone. This
    # isolates the loop's contribution.
    # For V1 (use_latent_feedback=False) this leaves nothing trainable -- pass
    # `--freeze_recurrent_blocks False --freeze_ln_f False --freeze_wte False`
    # to recover the V1 ablation surface (R blocks + ln_f + wte trainable).
    freeze_prelude: bool = True
    freeze_recurrent_blocks: bool = True
    freeze_coda: bool = True
    freeze_ln_f: bool = True
    freeze_wte: bool = True

    # ---- Diagnostics ----
    diag_log_every: int = 50


# Loop-param prefixes (after the HF base_model_prefix="model" wrapping).
_LOOP_PARAM_PREFIXES = (
    "model.recursive_link.",
    "model.damping_alpha_logit",
)
_LOOP_NO_DECAY_PARAM_NAMES = {"model.damping_alpha_logit"}


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

        self.loop_lr_mult = float(args.loop_lr_mult)
        self.diag_log_every = max(1, int(args.diag_log_every))

        self.freeze_prelude = bool(args.freeze_prelude)
        self.freeze_recurrent_blocks = bool(args.freeze_recurrent_blocks)
        self.freeze_coda = bool(args.freeze_coda)
        self.freeze_ln_f = bool(args.freeze_ln_f)
        self.freeze_wte = bool(args.freeze_wte)

        self._apply_block_freeze()

    # ------------------------------------------------------------------
    # Param freezing
    # ------------------------------------------------------------------

    def _apply_block_freeze(self) -> None:
        """Freeze backbone params per the configured profile.

        Defaults freeze every vanilla-LLaDA param (prelude + R + coda + ln_f
        + wte/ff_out), leaving ONLY recursive_link.* trainable. This isolates
        the loop's contribution and keeps optimizer state tiny.

        For V1 (use_latent_feedback=False) this default leaves nothing to
        train -- explicitly disable freeze_recurrent_blocks (and
        ln_f / wte if you want them trainable) on the V1 launch script.
        """
        if not (
            self.freeze_prelude or self.freeze_recurrent_blocks
            or self.freeze_coda or self.freeze_ln_f or self.freeze_wte
        ):
            return

        inner = self.model
        if hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "config") or not hasattr(inner, "transformer"):
            return

        cfg = inner.config
        P = int(getattr(cfg, "prelude_layers", 0))
        R = int(getattr(cfg, "recurrent_layers", 0))
        blocks = inner.transformer.blocks

        n_frozen_prelude = 0
        n_frozen_recurrent = 0
        n_frozen_coda = 0
        if self.freeze_prelude and P > 0:
            for block in blocks[:P]:
                for p in block.parameters():
                    p.requires_grad = False
                    n_frozen_prelude += p.numel()
        if self.freeze_recurrent_blocks and R > 0:
            for block in blocks[P:P + R]:
                for p in block.parameters():
                    p.requires_grad = False
                    n_frozen_recurrent += p.numel()
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
                for p in inner.transformer.ff_out.parameters():
                    p.requires_grad = False
                    n_frozen_wte += p.numel()

        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        try:
            from dllm.utils import get_default_logger
            logger = get_default_logger("mdlm_looped")
            logger.info(
                "[freeze] prelude=%d (%d params), recurrent_blocks=%d (%d params), "
                "coda=%d (%d params), ln_f=%d, wte=%d. trainable %d / %d (%.2f%%).",
                int(self.freeze_prelude), n_frozen_prelude,
                int(self.freeze_recurrent_blocks), n_frozen_recurrent,
                int(self.freeze_coda), n_frozen_coda,
                n_frozen_lnf, n_frozen_wte,
                n_train, n_total, 100.0 * n_train / max(1, n_total),
            )
            if n_train == 0:
                logger.warning(
                    "[freeze] No trainable parameters left. For V1 "
                    "(use_latent_feedback=False) you must override at least "
                    "freeze_recurrent_blocks=False."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Optimizer with loop-param LR multiplier
    # ------------------------------------------------------------------

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        loop_decay_params: List[nn.Parameter] = []
        loop_no_decay_params: List[nn.Parameter] = []

        decay_param_names = self.get_decay_parameter_names(self.model)

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_loop = _is_loop_param_name(name)
            is_decay = (
                name in decay_param_names
                and name not in _LOOP_NO_DECAY_PARAM_NAMES
            )
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
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    # ------------------------------------------------------------------
    # T_rec sampling
    # ------------------------------------------------------------------

    def _sample_t_rec(self, training: bool) -> int:
        if not training:
            return self.t_rec_eval
        if self.t_rec_min == self.t_rec_max:
            return self.t_rec_min
        return int(torch.randint(self.t_rec_min, self.t_rec_max + 1, ()).item())

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Mirror MDLMTrainer.compute_loss but pass T_rec and collect loop diagnostics."""

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

        # 3. Forward
        T_rec = self._sample_t_rec(training=model.training)
        want_diag = (
            getattr(self.state, "global_step", 0) % self.diag_log_every == 0
        ) and model.training

        outputs = model(
            input_ids=noised_input_ids,
            attention_mask=attention_mask,
            T_rec=T_rec,
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

        # 5. CE loss (final logits only)
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            input_ids,
            reduction="none",
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=logits.dtype).detach(),
        )

        # 6. Normalize
        if self.loss_norm_type == "token":
            token_nll = token_nll / maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll = token_nll / (
                maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
            )
        elif self.loss_norm_type == "batch":
            token_nll = token_nll / b
        else:
            raise ValueError("Invalid loss_norm_type.")

        loss = token_nll.sum()

        # 7. Diagnostic logging
        if want_diag and outputs.loop_diagnostics is not None:
            diag = dict(outputs.loop_diagnostics)
            self._log_loop_diagnostics(diag, T_rec=T_rec)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Diagnostic logging
    # ------------------------------------------------------------------

    def _log_loop_diagnostics(self, diag: Dict[str, Any], T_rec: int) -> None:
        log_dict: Dict[str, float] = {}

        log_dict["loop/T_rec"] = float(T_rec)

        for key in (
            "residual_norm",
            "raw_residual_norm",
            "adapter_update_norm",
            "damping_alpha",
        ):
            seq = diag.get(key, None)
            if not seq:
                continue
            for i, v in enumerate(seq):
                try:
                    log_dict[f"loop/{key}_iter{i}"] = float(v.item())
                except Exception:
                    pass

        for key in (
            "use_damped_update",
            "damping_schedule_id",
            "damping_tau",
            "damping_decay_beta",
            "learn_damping_alpha",
        ):
            value = diag.get(key, None)
            if value is None:
                continue
            try:
                if torch.is_tensor(value):
                    value = value.detach()
                log_dict[f"loop/{key}"] = float(value)
            except Exception:
                pass

        if log_dict:
            self.log(log_dict)
