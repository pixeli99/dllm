"""
MDLM-Looped trainer.

Extends MDLMTrainer for the latent-feedback looped LLaDA:

  1. Per-batch T_rec sampling (uniform over [t_rec_min, t_rec_max]).
     Same T_rec for the full batch (otherwise batched forward is impossible).

  2. 1-step truncated BPTT inside the model.

  3. Loop-param LR multiplier (recursive_link.*, workspace_slots, plus
     optional damping alpha/tau logits).
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
    # Loop params: recursive_link.* plus optional damping alpha/tau logits
    # plus workspace_slots.
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

    # ---- Deep supervision (v2) ----
    # When enabled, run intermediate loop-iteration hidden states through the
    # frozen coda + ln_f + tied lm_head, then add a weighted CE term on masked
    # positions. K iterations are randomly sampled per step from
    # {0, ..., T_rec - 2}; T_rec=1 contributes 0.
    enable_deep_sup: bool = False
    deep_sup_k_subset: int = 2
    lambda_deep: float = 0.3

    # ---- Diagnostics ----
    diag_log_every: int = 50


# Loop-param prefixes (after the HF base_model_prefix="model" wrapping).
_LOOP_PARAM_PREFIXES = (
    "model.recursive_link.",
    "model.damping_alpha_logit",
    "model.damping_tau_logit",
    "model.workspace_slots",
)
_LOOP_NO_DECAY_PARAM_NAMES = {
    "model.damping_alpha_logit",
    "model.damping_tau_logit",
    "model.workspace_slots",
}


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

        self.enable_deep_sup = bool(getattr(args, "enable_deep_sup", False))
        self.deep_sup_k_subset = max(1, int(getattr(args, "deep_sup_k_subset", 2)))
        self.lambda_deep = float(getattr(args, "lambda_deep", 0.3))

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

        want_loop_hiddens = (
            self.enable_deep_sup
            and self.lambda_deep > 0.0
            and model.training
            and T_rec > 1
        )

        outputs = model(
            input_ids=noised_input_ids,
            attention_mask=attention_mask,
            T_rec=T_rec,
            output_loop_diagnostics=want_diag,
            return_loop_hiddens=want_loop_hiddens,
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

        # 6b. Deep supervision (v2): coda-aware readout on masked positions.
        # No-op when disabled, when T_rec=1, or when no masked positions exist.
        if want_loop_hiddens and getattr(outputs, "loop_hiddens", None) is not None:
            deep_loss = self._compute_deep_supervision_loss(
                model=model,
                loop_hiddens=outputs.loop_hiddens,
                target_ids=input_ids,
                masked_mask=masked_mask,
                loss_weights=loss_weights,
                attention_mask=attention_mask,
            )
            loss = loss + self.lambda_deep * deep_loss
            if want_diag:
                try:
                    self.log({"loop/deep_sup_loss": float(deep_loss.detach().item())})
                except Exception:
                    pass

        # 7. Diagnostic logging
        if want_diag and outputs.loop_diagnostics is not None:
            diag = dict(outputs.loop_diagnostics)
            self._log_loop_diagnostics(diag, T_rec=T_rec)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Deep supervision helper (v2)
    # ------------------------------------------------------------------

    def _find_looped_inner(self, model):
        """Walk through DDP/HF wrappers until we reach LLaDALoopedModel."""
        inner = model
        # Bounded walk to avoid pathological loops.
        for _ in range(8):
            if hasattr(inner, "transformer") and hasattr(inner, "config"):
                return inner
            if hasattr(inner, "module"):
                inner = inner.module
            elif hasattr(inner, "model"):
                inner = inner.model
            else:
                break
        raise RuntimeError(
            "Could not locate LLaDALoopedModel in wrapper chain "
            f"starting at {type(model).__name__}; "
            "deep supervision needs access to .transformer.ln_f and .transformer.wte."
        )

    def _compute_deep_supervision_loss(
        self,
        model,
        loop_hiddens: List[torch.Tensor],
        target_ids: torch.Tensor,
        masked_mask: torch.Tensor,
        loss_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project intermediate loop hiddens through coda + LM readout.

        Memory: we gather masked positions BEFORE projecting to vocab so the
        per-iteration tensor is [N_masked, V] rather than [B, L, V]. With
        N_masked << B*L this avoids OOM on T_rec=6 BPTT.
        """
        inner = self._find_looped_inner(model)
        cfg = inner.config
        ln_f = inner.transformer.ln_f
        wte = inner.transformer.wte
        use_weight_tying = bool(getattr(cfg, "weight_tying", True))
        scale_logits = bool(getattr(cfg, "scale_logits", False))
        d_model = int(cfg.d_model)
        prelude_end = int(getattr(cfg, "prelude_layers", 0))
        coda_start = prelude_end + int(getattr(cfg, "recurrent_layers", 0))
        coda_blocks = inner.transformer.blocks[coda_start:]

        n_iter = len(loop_hiddens)
        # Final iteration's hidden flows through coda + main CE; supervise
        # only iterations 0..n_iter-2.
        n_inter = max(0, n_iter - 1)
        if n_inter == 0:
            return torch.zeros((), device=target_ids.device, dtype=loop_hiddens[0].dtype)

        masked_flat = masked_mask.bool()
        n_masked = int(masked_flat.sum().item())
        if n_masked == 0:
            return torch.zeros((), device=target_ids.device, dtype=loop_hiddens[0].dtype)

        k = min(self.deep_sup_k_subset, n_inter)
        perm = torch.randperm(n_inter, device=target_ids.device)
        indices = perm[:k].tolist()

        targets_flat = target_ids[masked_flat]  # [N]
        weights_flat = loss_weights[masked_flat]  # [N]
        norm = masked_mask.sum().clamp_min(1).to(dtype=loop_hiddens[0].dtype)
        batch = target_ids.shape[0]
        seq_len = target_ids.shape[1]

        coda_bias = None
        if attention_mask is not None and 0.0 in attention_mask:
            coda_bias = inner.get_bidirectional_attention_bias(
                seq_len, target_ids.device
            ).to(dtype=torch.float)
            coda_am = attention_mask.to(dtype=torch.float).view(
                batch, -1
            )[:, None, None, :]
            coda_am = (1.0 - coda_am) * torch.finfo(coda_am.dtype).min
            coda_bias = coda_bias[:, :, :seq_len, :seq_len] + coda_am

        deep_total = torch.zeros((), device=target_ids.device, dtype=loop_hiddens[0].dtype)
        for r in indices:
            h_r = loop_hiddens[r]  # [B, L, d]
            for block in coda_blocks:
                h_r, _ = block(
                    h_r,
                    attention_bias=coda_bias,
                    layer_past=None,
                    use_cache=False,
                )
            h_r = ln_f(h_r)
            h_r_masked = h_r[masked_flat]  # [N, d]
            if use_weight_tying:
                logits_r = F.linear(h_r_masked, wte.weight, None)
            else:
                logits_r = inner.transformer.ff_out(h_r_masked)
            if scale_logits:
                logits_r = logits_r * (1.0 / (d_model ** 0.5))

            ce_r = F.cross_entropy(logits_r, targets_flat, reduction="none")
            ce_r = ce_r * weights_flat.to(dtype=ce_r.dtype)

            if self.loss_norm_type == "batch":
                ce_r_sum = ce_r.sum() / batch
            else:
                # "token" (default) and "sequence" approximated as token-norm.
                ce_r_sum = ce_r.sum() / norm

            deep_total = deep_total + ce_r_sum

        return deep_total / float(len(indices))

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
            "workspace_norm",
            "workspace_update_norm",
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
            "learn_damping_tau",
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
