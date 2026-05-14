"""Dream-WoDD: WoDD retrofit on the Dream-7B diffusion LM (Phase 5).

WODD_PLAN.md §3.5 calls for a cross-family replication of the Phase 2
head-to-head on Dream-7B (GSM8K + MATH-500 only) to test whether the
WoDD effect transfers off LLaDA. We reuse the pipeline-agnostic
`_wodd_modules.py` (WriteHead / TapeRead / MemoryTape) directly; the
only new code is the Dream-specific config and the partitioned-forward
override.
"""
from .models.configuration_dream_wodd import DreamWoDDConfig
from .models.modeling_dream_wodd import DreamWoDDBaseModel, DreamWoDDModel

__all__ = [
    "DreamWoDDConfig",
    "DreamWoDDBaseModel",
    "DreamWoDDModel",
]
