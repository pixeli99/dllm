"""LLaDA-MetaState configuration.

State-evolution baseline (Phase 2 arm A2; Phase 3 capacity-sweep arm).
Matches LLaDA-WoDD's layer partition so Phase 2 holds FLOPs constant.
"""
from __future__ import annotations

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


class LLaDAMetaStateConfig(LLaDAConfig):
    model_type = "llada_metastate"

    @property
    def effective_n_kv_heads(self) -> int:
        return ModelConfig.effective_n_kv_heads.fget(self)

    def __init__(
        self,
        # ---- Layer partition (matched to WoDD) ----
        prelude_layers: int = 8,
        recurrent_layers: int = 16,
        coda_layers: int = 8,
        # ---- MetaState buffer ----
        # Inner thinking steps per denoising forward.
        t_step_eval: int = 4,
        # M : number of memory slots.  Phase 3 sweep: {64, 128, 256, 512, 1024}.
        state_slots: int = 64,
        # D : per-slot vector dim. Held at 1024 across Phase 3 to isolate M.
        state_dim: int = 1024,
        # Independent multi-head count for cross-attention `state_read`.
        state_n_heads: int = 8,
        # When True, the cross-attention's out_proj and the GRU's "z" gate
        # bias are configured so iteration 0 is bit-identical to a vanilla
        # LLaDA forward at T_step=1: state contribution to h is exactly 0
        # at init. Mirrors WoDD's `zero_init_wodd`.
        zero_init_metastate: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["LLaDAMetaStateModelLM"])
        super().__init__(**kwargs)

        self.prelude_layers   = int(prelude_layers)
        self.recurrent_layers = int(recurrent_layers)
        self.coda_layers      = int(coda_layers)

        self.t_step_eval      = int(t_step_eval)
        self.state_slots      = int(state_slots)
        self.state_dim        = int(state_dim)
        self.state_n_heads    = int(state_n_heads)
        self.zero_init_metastate = bool(zero_init_metastate)

        assert self.t_step_eval >= 1
        assert self.state_slots  >= 1
        assert self.state_dim    >= 1
        assert self.state_dim % self.state_n_heads == 0, (
            f"state_dim ({self.state_dim}) must be divisible by state_n_heads "
            f"({self.state_n_heads})"
        )
