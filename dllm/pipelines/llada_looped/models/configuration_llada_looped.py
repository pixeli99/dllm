"""
LLaDA-Looped configuration (latent feedback loop).

Extends LLaDAConfig with:
  - prelude/recurrent/coda layer partition
  - latent-feedback toggle (loop variant V2) vs pure h-recurrence (V1)
  - RecursiveMAS-style latent Adapter
  - optional damped recurrent state update for feedback iterations

V1 with T_rec=1 is numerically equivalent to vanilla LLaDA. V2 intentionally
passes the recurrent state through a trainable RecursiveLink before R, so it is
not a drop-in no-op.

Run:
    # This module is imported by the loaded model. See:
    #   /Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py
"""

from __future__ import annotations

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


class LLaDALoopedConfig(LLaDAConfig):
    model_type = "llada_looped"

    # `effective_n_kv_heads` is defined as a @property on ModelConfig, but
    # LLaDAConfig inherits from PretrainedConfig directly. Vanilla LLaDA
    # bridges via create_model_config_from_pretrained_config; we pass this
    # config straight through to LLaDAModel, so we need to expose the
    # property here.
    @property
    def effective_n_kv_heads(self) -> int:
        return ModelConfig.effective_n_kv_heads.fget(self)

    def __init__(
        self,
        # ---- Layer partition ----
        prelude_layers: int = 8,
        recurrent_layers: int = 16,
        coda_layers: int = 8,
        # ---- Loop variant toggle ----
        # V2 (use_latent_feedback=True): apply RecursiveLink (RecursiveMAS-style
        #     residual Adapter) to h_r before each recurrent sweep.
        # V1 (use_latent_feedback=False): pure h-recurrence h_{r+1} = R(h_r).
        # V0 baseline lives in examples/llada/sft.py (vanilla LLaDA, no loop).
        use_latent_feedback: bool = True,
        # ---- Eval-time T_rec default ----
        mu_rec_eval: int = 4,
        # ---- Optional feedback damping ----
        # When enabled, only feedback iterations (r >= 1) commit the proposed
        # recurrent state through h_next = h_prev + alpha * (proposal - h_prev).
        # Schedules:
        #   constant:   alpha = damping_alpha
        #   normalized: alpha = damping_tau / (T_rec - 1)
        #   decay:      sum_r alpha_r = damping_tau with exponential decay
        # learn_damping_tau makes damping_tau a learnable positive total budget
        # for normalized/decay schedules.
        # The first pass is never damped, preserving T_rec=1 equivalence.
        use_damped_update: bool = False,
        damping_schedule: str = "constant",
        damping_alpha: float = 0.5,
        learn_damping_alpha: bool = False,
        damping_tau: float = 1.0,
        learn_damping_tau: bool = True,
        damping_decay_beta: float = 0.0,
        # ---- Diffusion-conditioned damping ----
        # Optional DLM-specific controller for normalized/decay damping.
        # It keeps the same closed-loop fixed-point update, but lets the
        # total budget depend on current mask ratio and gates token updates
        # by whether the token is masked. Disabled by default.
        use_diffusion_conditioned_damping: bool = False,
        damping_masked_gate: float = 1.0,
        damping_unmasked_gate: float = 0.1,
        learn_mask_tau: bool = True,
        mask_tau_ref: float = 0.5,
        mask_tau_slope_init: float = 0.0,
        # ---- Feedback-only workspace slots ----
        # When workspace_size > 0, learnable continuous slots are appended only
        # during feedback recurrent passes (r >= 1). Prelude, first R pass, coda,
        # and LM head remain token-only. Therefore T_rec=1 is still exactly the
        # vanilla split path even when workspace_size > 0.
        workspace_size: int = 0,
        workspace_init_std: float = 0.02,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["LLaDALoopedModelLM"])
        super().__init__(**kwargs)

        # NB: validity of (prelude + recurrent + coda) <= n_layers is checked
        # in LLaDALoopedModel.__init__, not here. Reason: HF's to_diff_dict
        # serialization does `self.__class__()` with no kwargs and would
        # otherwise crash on n_layers=12 default during TensorBoard logging.

        self.prelude_layers = int(prelude_layers)
        self.recurrent_layers = int(recurrent_layers)
        self.coda_layers = int(coda_layers)

        self.use_latent_feedback = bool(use_latent_feedback)

        self.mu_rec_eval = int(mu_rec_eval)
        self.use_damped_update = bool(use_damped_update)
        self.damping_schedule = str(damping_schedule).lower()
        self.damping_alpha = float(damping_alpha)
        self.learn_damping_alpha = bool(learn_damping_alpha)
        self.damping_tau = float(damping_tau)
        self.learn_damping_tau = bool(learn_damping_tau)
        self.damping_decay_beta = float(damping_decay_beta)
        self.use_diffusion_conditioned_damping = bool(
            use_diffusion_conditioned_damping
        )
        self.damping_masked_gate = float(damping_masked_gate)
        self.damping_unmasked_gate = float(damping_unmasked_gate)
        self.learn_mask_tau = bool(learn_mask_tau)
        self.mask_tau_ref = float(mask_tau_ref)
        self.mask_tau_slope_init = float(mask_tau_slope_init)
        valid_schedules = {"constant", "normalized", "decay"}
        if self.damping_schedule not in valid_schedules:
            raise ValueError(
                f"damping_schedule must be one of {sorted(valid_schedules)}, "
                f"got {self.damping_schedule!r}"
            )
        if not 0.0 < self.damping_alpha <= 1.0:
            raise ValueError(
                f"damping_alpha must be in (0, 1], got {self.damping_alpha}"
            )
        if self.damping_tau <= 0.0:
            raise ValueError(f"damping_tau must be > 0, got {self.damping_tau}")
        if self.damping_decay_beta < 0.0:
            raise ValueError(
                f"damping_decay_beta must be >= 0, got {self.damping_decay_beta}"
            )
        if not 0.0 <= self.damping_unmasked_gate <= 1.0:
            raise ValueError(
                "damping_unmasked_gate must be in [0, 1], "
                f"got {self.damping_unmasked_gate}"
            )
        if not 0.0 <= self.damping_masked_gate <= 1.0:
            raise ValueError(
                "damping_masked_gate must be in [0, 1], "
                f"got {self.damping_masked_gate}"
            )
        if self.damping_masked_gate < self.damping_unmasked_gate:
            raise ValueError(
                "damping_masked_gate should be >= damping_unmasked_gate "
                "so masked tokens receive at least as much refinement."
            )
        if not 0.0 <= self.mask_tau_ref <= 1.0:
            raise ValueError(f"mask_tau_ref must be in [0, 1], got {self.mask_tau_ref}")
        if self.learn_damping_alpha and self.damping_schedule != "constant":
            raise ValueError(
                "learn_damping_alpha is only supported for constant damping_schedule"
            )

        self.workspace_size = int(workspace_size)
        self.workspace_init_std = float(workspace_init_std)
        if self.workspace_size < 0:
            raise ValueError(
                f"workspace_size must be >= 0, got {self.workspace_size}"
            )
        if self.workspace_init_std <= 0.0:
            raise ValueError(
                f"workspace_init_std must be > 0, got {self.workspace_init_std}"
            )
