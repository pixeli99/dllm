"""Config for LLaDA-LATTS (latent test-time steps; no memory).

Layer partition mirrors LLaDA-WoDD's so the comparison in Phase 2 holds
FLOPs constant:
  - prelude_layers  blocks run once
  - recurrent_layers blocks run t_step_eval times (weight-shared)
  - coda_layers      blocks run once

LATTS has *no* WoDD modules and *no* persistent state; the only knob
beyond vanilla LLaDA is `t_step_eval`. Phase 2's matched-FLOPs setting
uses t_step_eval=4 so this arm spends the same compute as the recurrent
arm in WoDD / MetaState.
"""
from __future__ import annotations

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


class LLaDALATTSConfig(LLaDAConfig):
    model_type = "llada_latts"

    @property
    def effective_n_kv_heads(self) -> int:
        # Mirror LLaDAWoDDConfig: HF's PretrainedConfig.__getattribute__
        # short-circuits dataclass-style derived properties, so we have to
        # bridge to the base ModelConfig descriptor explicitly.
        return ModelConfig.effective_n_kv_heads.fget(self)

    def __init__(
        self,
        prelude_layers: int = 8,
        recurrent_layers: int = 16,
        coda_layers: int = 8,
        t_step_eval: int = 4,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["LLaDALATTSModelLM"])
        super().__init__(**kwargs)
        # WODD_PLAN.md §3.2 / §3.3: matched layer partition across all arms.
        self.prelude_layers   = int(prelude_layers)
        self.recurrent_layers = int(recurrent_layers)
        self.coda_layers      = int(coda_layers)
        # Inner-loop iterations of the recurrent block. T_step=1 reduces
        # to vanilla LLaDA (no extra compute).
        self.t_step_eval      = int(t_step_eval)
        assert self.t_step_eval >= 1
