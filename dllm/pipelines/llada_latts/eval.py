"""LLaDA-LATTS eval entry (Phase 2 / 3 baseline).

Identical wiring to dllm/pipelines/llada/eval.py except registered as
`llada_latts` so lm-eval can dispatch by `--model llada_latts`. The
underlying loglikelihood / generate_until paths are inherited unchanged
from MDLMEvalHarness; the model class is auto-resolved via
AutoConfig/AutoModel registration in the pipeline's models/__init__.py.

Run example:
    accelerate launch \\
        --num_processes 4 \\
        dllm/pipelines/llada_latts/eval.py \\
        --tasks gsm8k_cot --model llada_latts \\
        --apply_chat_template --num_fewshot 5 \\
        --model_args "pretrained=PATH_TO_CHECKPOINT,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0"
"""
from __future__ import annotations

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

# Import the pipeline so AutoConfig.register("llada_latts", ...) fires
# before any from_pretrained call.
from dllm.pipelines.llada_latts import LLaDALATTSConfig, LLaDALATTSModelLM  # noqa: F401


@dataclass
class LLaDALATTSEvalSamplerConfig(MDLMSamplerConfig):
    """Default sampler config matching LLaDA's eval recipe."""
    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class LLaDALATTSEvalConfig(MDLMEvalConfig):
    """LATTS eval config; LLaDA-8B-Base's max_length."""
    max_length: int = 4096


@register_model("llada_latts")
class LLaDALATTSEvalHarness(MDLMEvalHarness):
    """LATTS uses the model's `config.t_step_eval` (default 4) for the
    inner loop, so the standard _get_logits(batch) path through forward()
    already applies T_step. No override needed."""

    def __init__(
        self,
        eval_config: LLaDALATTSEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDALATTSEvalConfig()
        sampler_config = sampler_config or LLaDALATTSEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
