"""LLaDA-WoDD eval entry (Phase 2 main arm + Phase 3 sweep).

Mirrors dllm/pipelines/llada/eval.py. T_step comes from the model's
config.t_step_eval; the standard forward path handles it.
"""
from __future__ import annotations

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

# Trigger AutoConfig.register("llada_wodd", ...) early.
from dllm.pipelines.llada_wodd import LLaDAWoDDConfig, LLaDAWoDDModelLM  # noqa: F401


@dataclass
class LLaDAWoDDEvalSamplerConfig(MDLMSamplerConfig):
    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class LLaDAWoDDEvalConfig(MDLMEvalConfig):
    max_length: int = 4096


@register_model("llada_wodd")
class LLaDAWoDDEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: LLaDAWoDDEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDAWoDDEvalConfig()
        sampler_config = sampler_config or LLaDAWoDDEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
