"""LLaDA-MetaState: state-evolution baseline (A2).

Reference: arXiv 2603.01331 ("MetaState").
A persistent fixed-capacity memory of M slots × D dims, updated IN PLACE
each inner step by a GRU-style gate that overwrites old contents with a
function of the new pooled hidden state. This is the "state evolution"
shape WoDD's append-only argument is contrasted against (WODD_PLAN.md
§§1.2-1.3).

The capacity ceiling that the type-of-memory hypothesis predicts:
no matter how many inner steps the model takes, only ~ M·D bits survive
the GRU's contraction. WoDD's pre-registered scaling-law prediction in
§3.3 says MetaState saturates by M=1024, WoDD does not.
"""

from .models.configuration_llada_metastate import LLaDAMetaStateConfig
from .models.modeling_llada_metastate import (
    LLaDAMetaStateModel, LLaDAMetaStateModelLM,
)

__all__ = [
    "LLaDAMetaStateConfig",
    "LLaDAMetaStateModel",
    "LLaDAMetaStateModelLM",
]
