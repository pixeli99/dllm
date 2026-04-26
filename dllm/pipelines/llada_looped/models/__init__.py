"""
LLaDA-Looped model package.

Registers `llada_looped` with HF's AutoConfig / AutoModel / AutoModelForMaskedLM
so that `import dllm` exposes it end-to-end.
"""

from .configuration_llada_looped import LLaDALoopedConfig
from .modeling_llada_looped import LLaDALoopedModel, LLaDALoopedModelLM

__all__ = [
    "LLaDALoopedConfig",
    "LLaDALoopedModel",
    "LLaDALoopedModelLM",
]

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada_looped", LLaDALoopedConfig)
    AutoModel.register(LLaDALoopedConfig, LLaDALoopedModelLM)
    AutoModelForMaskedLM.register(LLaDALoopedConfig, LLaDALoopedModelLM)
except (ImportError, ValueError):
    # ValueError can happen on re-import (duplicate registration).
    pass
