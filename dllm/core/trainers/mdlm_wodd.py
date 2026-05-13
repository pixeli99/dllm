"""
MDLM-WoDD trainer.

Extends MDLMTrainer for the Write-on-Demand Diffusion LLaDA variant. Three
additions on top of vanilla MDLM:

  1. Per-batch T_step sampling (uniform over [t_step_min, t_step_max]).
     Sampled once per batch so the forward stays batched.

  2. Three-component loss:
         L = L_mdlm + λ_sparsity * mean(g_k) - λ_entropy * mean(H(Bern(g_k)))
     - sparsity: penalize total write rate so the model only writes when
       useful. λ_sparsity > 0.
     - entropy:  reward gate entropy so it doesn't collapse to always-off
       or always-on. We SUBTRACT entropy (so loss decreases as entropy
       rises). λ_entropy > 0.

  3. LR multiplier for WoDD params (write_head.*, tape_read.*).

  4. Trainable-surface control mirroring MDLMLoopedConfig: by default freeze
     the vanilla LLaDA backbone, train only the WoDD-specific modules.

  5. Diagnostics:
         wodd/T_step
         wodd/mean_gate
         wodd/gate_entropy
         wodd/tape_contrib_rel_norm_iter{k}
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
class MDLMWoDDConfig(MDLMConfig):
    # ---- T_step (inner thinking steps) sampling at training time ----
    t_step_min: int = 1
    t_step_max: int = 4
    # Eval-time T_step (also written into model config).
    t_step_eval: int = 4

    # ---- WoDD-specific loss coefficients ----
    # write_sparsity_lambda * mean(g_k) penalizes writing -- model only writes
    # when the MDLM CE improvement is worth it.
    write_sparsity_lambda: float = 1e-2
    # entropy_reg_lambda * mean(H(Bern(g_k))). We SUBTRACT this from loss so
    # higher entropy is rewarded. Prevents gate collapse to always-on / -off.
    gate_entropy_lambda: float = 1e-3

    # ---- LR multiplier for WoDD params ----
    wodd_lr_mult: float = 10.0

    # ---- Trainable-surface control ----
    freeze_prelude: bool = True
    freeze_recurrent_blocks: bool = True
    freeze_coda: bool = True
    freeze_ln_f: bool = True
    freeze_wte: bool = True

    # ---- Diagnostics ----
    diag_log_every: int = 50


_WODD_PARAM_PREFIXES = (
    "model.tape_read.",
    "model.write_head.",
)


def _is_wodd_param_name(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in _WODD_PARAM_PREFIXES)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MDLMWoDDTrainer(MDLMTrainer):

    def __init__(self, args: MDLMWoDDConfig, *pargs, **kwargs):
        super().__init__(args=args, *pargs, **kwargs)

        self.t_step_min = int(args.t_step_min)
        self.t_step_max = int(args.t_step_max)
        self.t_step_eval = int(args.t_step_eval)
        assert self.t_step_min >= 1
        assert self.t_step_max >= self.t_step_min

        self.write_sparsity_lambda = float(args.write_sparsity_lambda)
        self.gate_entropy_lambda = float(args.gate_entropy_lambda)
        self.wodd_lr_mult = float(args.wodd_lr_mult)
        self.diag_log_every = max(1, int(args.diag_log_every))

        self.freeze_prelude = bool(args.freeze_prelude)
        self.freeze_recurrent_blocks = bool(args.freeze_recurrent_blocks)
        self.freeze_coda = bool(args.freeze_coda)
        self.freeze_ln_f = bool(args.freeze_ln_f)
        self.freeze_wte = bool(args.freeze_wte)

        self._apply_block_freeze()

    # ------------------------------------------------------------------
    # Param freezing (parallels MDLMLoopedTrainer._apply_block_freeze)
    # ------------------------------------------------------------------

    def _apply_block_freeze(self) -> None:
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

        n_frozen = {"prelude": 0, "recurrent": 0, "coda": 0, "ln_f": 0, "wte": 0}

        if self.freeze_prelude and P > 0:
            for block in blocks[:P]:
                for p in block.parameters():
                    p.requires_grad = False
                    n_frozen["prelude"] += p.numel()
        if self.freeze_recurrent_blocks and R > 0:
            for block in blocks[P:P + R]:
                for p in block.parameters():
                    p.requires_grad = False
                    n_frozen["recurrent"] += p.numel()
        if self.freeze_coda:
            for block in blocks[P + R:]:
                for p in block.parameters():
                    p.requires_grad = False
                    n_frozen["coda"] += p.numel()
        if self.freeze_ln_f and hasattr(inner.transformer, "ln_f"):
            for p in inner.transformer.ln_f.parameters():
                p.requires_grad = False
                n_frozen["ln_f"] += p.numel()
        if self.freeze_wte and hasattr(inner.transformer, "wte"):
            for p in inner.transformer.wte.parameters():
                p.requires_grad = False
                n_frozen["wte"] += p.numel()
            if hasattr(inner.transformer, "ff_out"):
                for p in inner.transformer.ff_out.parameters():
                    p.requires_grad = False
                    n_frozen["wte"] += p.numel()

        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        try:
            from dllm.utils import get_default_logger
            logger = get_default_logger("mdlm_wodd")
            logger.info(
                "[wodd freeze] prelude=%d recurrent=%d coda=%d ln_f=%d wte=%d; "
                "trainable %d / %d (%.2f%%)",
                n_frozen["prelude"], n_frozen["recurrent"], n_frozen["coda"],
                n_frozen["ln_f"], n_frozen["wte"],
                n_train, n_total, 100.0 * n_train / max(1, n_total),
            )
            if n_train == 0:
                logger.warning(
                    "[wodd freeze] No trainable parameters left."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Optimizer with WoDD-param LR multiplier
    # ------------------------------------------------------------------

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        wodd_decay_params: List[nn.Parameter] = []
        wodd_no_decay_params: List[nn.Parameter] = []

        decay_param_names = self.get_decay_parameter_names(self.model)

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_wodd = _is_wodd_param_name(name)
            is_decay = name in decay_param_names
            if is_wodd:
                (wodd_decay_params if is_decay else wodd_no_decay_params).append(p)
            else:
                (decay_params if is_decay else no_decay_params).append(p)

        base_lr = self.args.learning_rate
        wodd_lr = base_lr * self.wodd_lr_mult
        wd = self.args.weight_decay

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "lr": base_lr, "weight_decay": wd})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0})
        if wodd_decay_params:
            param_groups.append({"params": wodd_decay_params, "lr": wodd_lr, "weight_decay": wd})
        if wodd_no_decay_params:
            param_groups.append({"params": wodd_no_decay_params, "lr": wodd_lr, "weight_decay": 0.0})

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    # ------------------------------------------------------------------
    # T_step sampling
    # ------------------------------------------------------------------

    def _sample_t_step(self, training: bool) -> int:
        if not training:
            return self.t_step_eval
        if self.t_step_min == self.t_step_max:
            return self.t_step_min
        return int(torch.randint(self.t_step_min, self.t_step_max + 1, ()).item())

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """MDLM CE + write_sparsity + gate_entropy.

        L = L_mdlm + λ_s * mean(g_k) - λ_e * mean(H(Bern(g_k)))
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100

        # 1. Diffusion timesteps
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. Stochastic masking
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # 3. Forward
        T_step = self._sample_t_step(training=model.training)
        want_diag = (
            getattr(self.state, "global_step", 0) % self.diag_log_every == 0
        ) and model.training

        outputs = model(
            input_ids=noised_input_ids,
            attention_mask=attention_mask,
            T_step=T_step,
            output_tape_diagnostics=want_diag,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # 4. MDLM CE
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

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

        loss_mdlm = token_nll.sum()

        # 5. WoDD regularizers
        # write_gates: (B, T_step, 1)
        gates = outputs.write_gates
        gates_flat = gates.squeeze(-1).clamp(min=1e-6, max=1.0 - 1e-6)  # (B, T_step)

        # Sparsity: penalize mean gate value. Differentiable.
        loss_sparsity = gates_flat.mean()

        # Entropy of each Bernoulli(g_k), averaged. Want this high.
        bernoulli_entropy = -(
            gates_flat * torch.log(gates_flat)
            + (1 - gates_flat) * torch.log(1 - gates_flat)
        )
        loss_entropy = bernoulli_entropy.mean()

        loss = (
            loss_mdlm
            + self.write_sparsity_lambda * loss_sparsity
            - self.gate_entropy_lambda * loss_entropy
        )

        # 6. Diagnostics
        if want_diag:
            self._log_wodd_diagnostics(
                T_step=T_step,
                mean_gate=gates_flat.mean().detach(),
                gate_entropy=bernoulli_entropy.mean().detach(),
                loss_mdlm=loss_mdlm.detach(),
                loss_sparsity=loss_sparsity.detach(),
                loss_entropy=loss_entropy.detach(),
                tape_diagnostics=getattr(outputs, "tape_diagnostics", None),
            )

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Diagnostic logging
    # ------------------------------------------------------------------

    def _log_wodd_diagnostics(
        self,
        T_step: int,
        mean_gate: torch.Tensor,
        gate_entropy: torch.Tensor,
        loss_mdlm: torch.Tensor,
        loss_sparsity: torch.Tensor,
        loss_entropy: torch.Tensor,
        tape_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        log_dict: Dict[str, float] = {
            "wodd/T_step": float(T_step),
            "wodd/mean_gate": float(mean_gate.item()),
            "wodd/gate_entropy": float(gate_entropy.item()),
            "wodd/loss_mdlm": float(loss_mdlm.item()),
            "wodd/loss_sparsity": float(loss_sparsity.item()),
            "wodd/loss_entropy": float(loss_entropy.item()),
        }
        if tape_diagnostics is not None:
            norms = tape_diagnostics.get("tape_contrib_rel_norm", []) or []
            for i, v in enumerate(norms):
                try:
                    log_dict[f"wodd/tape_contrib_rel_norm_iter{i}"] = float(v.item())
                except Exception:
                    pass
        if log_dict:
            self.log(log_dict)
