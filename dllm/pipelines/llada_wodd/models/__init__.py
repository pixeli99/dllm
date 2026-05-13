from .configuration_llada_wodd import LLaDAWoDDConfig
from .modeling_llada_wodd import LLaDAWoDDModel, LLaDAWoDDModelLM

__all__ = [
    "LLaDAWoDDConfig",
    "LLaDAWoDDModel",
    "LLaDAWoDDModelLM",
]

# Register with HuggingFace Auto classes so a config.json with
# `model_type: "llada_wodd"` resolves on `from_pretrained`.
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada_wodd", LLaDAWoDDConfig)
    AutoModel.register(LLaDAWoDDConfig, LLaDAWoDDModelLM)
    AutoModelForMaskedLM.register(LLaDAWoDDConfig, LLaDAWoDDModelLM)
except ImportError:
    pass
