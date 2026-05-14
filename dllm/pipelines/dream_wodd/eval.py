"""Dream-WoDD eval entry (Phase 5 cross-family).

Mirrors dllm/pipelines/llada_wodd/eval.py. Registers `dream_wodd` with
lm_eval for GSM8K / MATH-500 cross-family replication per WODD_PLAN §3.5.
"""
from __future__ import annotations

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

# Trigger AutoConfig.register("dream_wodd", ...) at import.
from dllm.pipelines.dream_wodd import DreamWoDDConfig, DreamWoDDModel  # noqa: F401


@dataclass
class DreamWoDDEvalSamplerConfig(MDLMSamplerConfig):
    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class DreamWoDDEvalConfig(MDLMEvalConfig):
    max_length: int = 4096


@register_model("dream_wodd")
class DreamWoDDEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: DreamWoDDEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or DreamWoDDEvalConfig()
        sampler_config = sampler_config or DreamWoDDEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
