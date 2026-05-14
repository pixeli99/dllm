from .configuration_llada_metastate import LLaDAMetaStateConfig
from .modeling_llada_metastate import LLaDAMetaStateModel, LLaDAMetaStateModelLM

__all__ = [
    "LLaDAMetaStateConfig",
    "LLaDAMetaStateModel",
    "LLaDAMetaStateModelLM",
]

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada_metastate", LLaDAMetaStateConfig)
    AutoModel.register(LLaDAMetaStateConfig, LLaDAMetaStateModelLM)
    AutoModelForMaskedLM.register(LLaDAMetaStateConfig, LLaDAMetaStateModelLM)
except ImportError:
    pass
