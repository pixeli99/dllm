from .configuration_llada_looped import LLaDALoopedConfig
from .modeling_llada_looped import LLaDALoopedModel, LLaDALoopedModelLM

__all__ = [
    "LLaDALoopedConfig",
    "LLaDALoopedModel",
    "LLaDALoopedModelLM",
]

# Register with HuggingFace Auto classes so a config.json with
# `model_type: "llada_looped"` resolves to our class on `from_pretrained`.
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada_looped", LLaDALoopedConfig)
    AutoModel.register(LLaDALoopedConfig, LLaDALoopedModelLM)
    AutoModelForMaskedLM.register(LLaDALoopedConfig, LLaDALoopedModelLM)

except ImportError:
    pass
