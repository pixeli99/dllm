"""LLaDA-DPad eval entry (Phase 2 baseline).

DPad has no inner-loop T_step; it runs the standard 32-block LLaDA
forward once over [seq | scratchpad]. The model class handles the
scratchpad attach + post-forward strip internally.

The MDLM sampler operates on the original-positions logits because
LLaDADPadModel.forward strips the K scratchpad positions before
projecting to vocab.
"""
from __future__ import annotations

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

from dllm.pipelines.llada_dpad import (
    LLaDADPadConfig, LLaDADPadModelLM,  # noqa: F401
)


@dataclass
class LLaDADPadEvalSamplerConfig(MDLMSamplerConfig):
    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class LLaDADPadEvalConfig(MDLMEvalConfig):
    max_length: int = 4096


@register_model("llada_dpad")
class LLaDADPadEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: LLaDADPadEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDADPadEvalConfig()
        sampler_config = sampler_config or LLaDADPadEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
