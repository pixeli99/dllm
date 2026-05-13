from .configuration_llada_latts import LLaDALATTSConfig
from .modeling_llada_latts import LLaDALATTSModel, LLaDALATTSModelLM

__all__ = [
    "LLaDALATTSConfig",
    "LLaDALATTSModel",
    "LLaDALATTSModelLM",
]

# Register with HuggingFace Auto classes so a config.json with
# `model_type: "llada_latts"` resolves on `from_pretrained`.
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada_latts", LLaDALATTSConfig)
    AutoModel.register(LLaDALATTSConfig, LLaDALATTSModelLM)
    AutoModelForMaskedLM.register(LLaDALATTSConfig, LLaDALATTSModelLM)
except ImportError:
    pass
