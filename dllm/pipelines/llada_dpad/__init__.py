"""LLaDA-DPad: suffix-scratch baseline (A3).

Reference: arXiv 2508.14148 ("DPad: Diffusion Padding").
The model prepends/appends K learnable "scratchpad" positions to the
sequence; the standard LLaDA self-attention treats them as ordinary
tokens, updating them at every layer. After the forward, the scratchpad
positions are stripped from the logits.

There is NO per-step writing and NO append-only behaviour -- the
scratchpad is updated in place by self-attention. T_step=1.

This is the "augmented context" baseline. Useful contrast against WoDD:
DPad gives the model extra token-space capacity but does not change the
type of memory.
"""

from .models.configuration_llada_dpad import LLaDADPadConfig
from .models.modeling_llada_dpad import LLaDADPadModel, LLaDADPadModelLM

__all__ = [
    "LLaDADPadConfig",
    "LLaDADPadModel",
    "LLaDADPadModelLM",
]
