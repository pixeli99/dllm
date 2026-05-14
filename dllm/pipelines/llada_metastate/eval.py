"""LLaDA-MetaState eval entry (Phase 2 / 3 baseline).

Mirrors dllm/pipelines/llada/eval.py with the MetaState pipeline as the
backing model. Inner-loop T_step is read from the model config; no
override needed in this harness.
"""
from __future__ import annotations

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

# Trigger AutoConfig.register("llada_metastate", ...) early.
from dllm.pipelines.llada_metastate import (
    LLaDAMetaStateConfig, LLaDAMetaStateModelLM,  # noqa: F401
)


@dataclass
class LLaDAMetaStateEvalSamplerConfig(MDLMSamplerConfig):
    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class LLaDAMetaStateEvalConfig(MDLMEvalConfig):
    max_length: int = 4096


@register_model("llada_metastate")
class LLaDAMetaStateEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: LLaDAMetaStateEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDAMetaStateEvalConfig()
        sampler_config = sampler_config or LLaDAMetaStateEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
