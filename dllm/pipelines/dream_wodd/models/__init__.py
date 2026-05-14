from .configuration_dream_wodd import DreamWoDDConfig
from .modeling_dream_wodd import DreamWoDDBaseModel, DreamWoDDModel

__all__ = ["DreamWoDDConfig", "DreamWoDDBaseModel", "DreamWoDDModel"]

try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register("dream_wodd", DreamWoDDConfig)
    AutoModel.register(DreamWoDDConfig, DreamWoDDModel)
    AutoModelForCausalLM.register(DreamWoDDConfig, DreamWoDDModel)
except ImportError:
    pass
