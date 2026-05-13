"""LLaDA-LATTS: latent test-time steps with NO memory.

Baseline A1 from WODD_PLAN.md §3.2 — the "iteration without memory" arm.
The model runs the same partitioned forward as WoDD (prelude → recurrent →
coda) but iterates the recurrent block T_step times with no read/write
adapter at all. Trainable params: 0 (the backbone is frozen; nothing else
exists to learn). T_step is purely an inference-time knob.

Purpose: isolate the contribution of "extra compute" from "extra memory".
If LATTS matches MetaState/WoDD on math reasoning, the type-of-memory
claim is falsified at this scale and the paper writes the negative result.
"""

from .models.configuration_llada_latts import LLaDALATTSConfig
from .models.modeling_llada_latts import LLaDALATTSModel, LLaDALATTSModelLM

__all__ = [
    "LLaDALATTSConfig",
    "LLaDALATTSModel",
    "LLaDALATTSModelLM",
]
