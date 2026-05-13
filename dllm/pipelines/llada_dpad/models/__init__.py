from .configuration_llada_dpad import LLaDADPadConfig
from .modeling_llada_dpad import LLaDADPadModel, LLaDADPadModelLM

__all__ = [
    "LLaDADPadConfig",
    "LLaDADPadModel",
    "LLaDADPadModelLM",
]

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada_dpad", LLaDADPadConfig)
    AutoModel.register(LLaDADPadConfig, LLaDADPadModelLM)
    AutoModelForMaskedLM.register(LLaDADPadConfig, LLaDADPadModelLM)
except ImportError:
    pass
