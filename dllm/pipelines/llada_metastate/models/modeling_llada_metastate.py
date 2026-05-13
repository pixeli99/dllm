"""LLaDA-MetaState modeling.

State-evolution baseline for Phase 2 / 3. Architecture per inner step k:

    state = state_init (broadcast over batch)
    for k in 1..T_step:
        # state -> sequence
        h = h + state_read(h, state)
        # recurrent block over the sequence (weight-shared across k)
        h = recurrent_blocks(h)
        # sequence -> state (IN-PLACE overwrite, not append)
        pool = masked_mean(h)
        state = state_update(state, pool)        # GRU update
    coda(h) -> logits

Invariants:
  - With `zero_init_metastate=True`, iteration 0 is bit-identical to a
    vanilla LLaDA forward at T_step=1: state_read.out_proj==0 makes the
    cross-attention contribute 0, and the GRU's z-bias=-5 makes
    state'~=state so the buffer is unchanged.
  - State is overwritten in place at every k (THIS IS THE POINT relative
    to WoDD's append-only tape; see WODD_PLAN.md §1.2 information-
    theoretic argument).
  - The number of slots M is a config knob, swept in Phase 3.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from dllm.pipelines.llada.models.modeling_llada import (
    LLaDAModel,
    LLaDAModelLM,
    LLaDAOutput,
    LLaDAPreTrainedModel,
    ensure_finite_,
)

from ._metastate_modules import (
    MetaStateGRU, MetaStateInit, StateRead, masked_mean,
)
from .configuration_llada_metastate import LLaDAMetaStateConfig


@dataclass
class MetaStateOutput(CausalLMOutputWithPast):
    """LLaDA output extended with a final-state diagnostic.

    `final_state_rel_norm` is the L2 norm of state at the end of T_step
    iterations, normalised by the initial-state norm -- a coarse measure
    of "how much was overwritten". For the capacity-ceiling argument in
    WODD_PLAN.md §1.2 we expect this to plateau as M grows.
    """
    final_state_rel_norm: Optional[torch.Tensor] = None


class LLaDAMetaStateModel(LLaDAModel):
    config_class = LLaDAMetaStateConfig

    def __init__(self, config: LLaDAMetaStateConfig, init_params: bool = True):
        super().__init__(config, init_params=init_params)

        total = (
            int(config.prelude_layers)
            + int(config.recurrent_layers)
            + int(config.coda_layers)
        )
        assert total <= config.n_layers, (
            f"prelude({config.prelude_layers}) + recurrent({config.recurrent_layers}) "
            f"+ coda({config.coda_layers}) = {total} > n_layers ({config.n_layers})"
        )
        assert int(config.recurrent_layers) >= 1
        assert int(config.coda_layers)      >= 1

        D = int(config.state_dim)
        d = int(config.d_model)
        dev = config.init_device

        self.state_init = MetaStateInit(
            state_slots=config.state_slots,
            state_dim=D,
            zero_init=config.zero_init_metastate,
            init_device=dev,
        )
        self.state_read = StateRead(
            d_model=d,
            state_dim=D,
            n_heads=config.state_n_heads,
            zero_init=config.zero_init_metastate,
            init_device=dev,
        )
        self.state_update = MetaStateGRU(
            d_model=d,
            state_dim=D,
            zero_init=config.zero_init_metastate,
            init_device=dev,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        T_step: Optional[int] = None,
        output_state_diagnostics: bool = False,
    ) -> Tuple[LLaDAOutput, Dict[str, Any]]:
        assert not self.config.alibi
        assert self.config.rope
        assert past_key_values is None and not use_cache
        assert self.config.block_group_size == 1

        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False

        if T_step is None:
            T_step = int(self.config.t_step_eval)
        T_step = max(1, int(T_step))

        if input_embeddings is not None:
            B_, T_, _ = input_embeddings.size()
            x = input_embeddings
        else:
            B_, T_ = input_ids.size()
            x = self.transformer.wte(input_ids)

        if self.config.input_emb_norm:
            x = x * (self.config.d_model ** 0.5)
        x = self.transformer.emb_drop(x)

        if attention_mask is not None and 0.0 in attention_mask:
            am = attention_mask.to(dtype=torch.float).view(B_, -1)[:, None, None, :]
            am = (1.0 - am) * torch.finfo(am.dtype).min
        else:
            am = None

        eff_bias: Optional[torch.Tensor] = None
        if attention_bias is not None or am is not None:
            if attention_bias is None:
                eff_bias = self.get_bidirectional_attention_bias(T_, x.device)
            else:
                eff_bias = attention_bias
                if eff_bias.dtype in (torch.int8, torch.bool):
                    eff_bias = eff_bias.to(dtype=torch.float)
                    eff_bias.masked_fill_(eff_bias == 0.0, torch.finfo(eff_bias.dtype).min)
            mask_len = T_ if am is None else am.shape[-1]
            eff_bias = eff_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            if am is not None:
                eff_bias = eff_bias + am
                ensure_finite_(eff_bias, check_neg_inf=True, check_pos_inf=False)

        P = int(self.config.prelude_layers)
        R = int(self.config.recurrent_layers)
        coda_start = P + R
        blocks = self.transformer.blocks
        all_hidden_states: List[torch.Tensor] = []

        # Prelude
        for block in blocks[:P]:
            if output_hidden_states:
                all_hidden_states.append(x)
            x, _ = block(x, attention_bias=eff_bias, layer_past=None, use_cache=False)

        # Init MetaState buffer.
        # Cast to the same dtype as the activations for clean bf16/fp16 training.
        state = self.state_init(B_).to(dtype=x.dtype)
        initial_norm = None
        if output_state_diagnostics:
            with torch.no_grad():
                initial_norm = state.norm()

        # Inner loop
        h = x
        for _k in range(T_step):
            h = h + self.state_read(h, state, attention_mask=attention_mask)
            for block in blocks[P:coda_start]:
                h, _ = block(h, attention_bias=eff_bias, layer_past=None, use_cache=False)
            pool = masked_mean(h, attention_mask)
            state = self.state_update(state, pool)        # IN-PLACE GRU update

        # Coda
        x = h
        for block in blocks[coda_start:]:
            if output_hidden_states:
                all_hidden_states.append(x)
            x, _ = block(x, attention_bias=eff_bias, layer_past=None, use_cache=False)

        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)

        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits = logits * (1.0 / math.sqrt(self.config.d_model))

        out = LLaDAOutput(
            logits=logits,
            attn_key_values=None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )

        diagnostics: Dict[str, Any] = {"T_step": T_step}
        if output_state_diagnostics and initial_norm is not None:
            with torch.no_grad():
                diagnostics["final_state_rel_norm"] = (
                    state.norm() / initial_norm.clamp_min(1e-6)
                )
        return out, diagnostics


class LLaDAMetaStateModelLM(LLaDAModelLM):
    config_class = LLaDAMetaStateConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDALlamaBlock"]

    def __init__(
        self,
        config: LLaDAMetaStateConfig,
        model: Optional[LLaDAMetaStateModel] = None,
        init_params: bool = False,
    ):
        LLaDAPreTrainedModel.__init__(self, config)
        if model is None:
            config.init_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = LLaDAMetaStateModel(config, init_params=init_params)
        else:
            self.model = model

    @classmethod
    def from_llada_checkpoint(
        cls,
        model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        **metastate_kwargs,
    ) -> "LLaDAMetaStateModelLM":
        from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
        from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        base_config = LLaDAConfig.from_pretrained(model_name_or_path)
        config_dict = base_config.to_dict()
        config_dict.pop("model_type", None)
        config_dict.pop("architectures", None)
        config_dict.update(metastate_kwargs)
        metastate_config = cls.config_class(**config_dict)

        vanilla = LLaDAModelLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        )

        model = cls(metastate_config, init_params=False)
        model = model.to(dtype=torch_dtype)

        missing, unexpected = model.load_state_dict(
            vanilla.state_dict(), strict=False
        )
        if unexpected:
            import warnings
            warnings.warn(
                f"Unexpected keys loading vanilla LLaDA into MetaState: "
                f"{list(unexpected)[:8]}", UserWarning,
            )

        del vanilla
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        T_step: Optional[int] = None,
        output_state_diagnostics: bool = False,
    ) -> Union[Tuple, MetaStateOutput]:
        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not supported in LLaDA-MetaState.")
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs, diagnostics = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            T_step=T_step,
            output_state_diagnostics=output_state_diagnostics,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        if labels is not None:
            import warnings
            warnings.warn(
                "LLaDA-MetaState does not compute loss inside forward; use the MDLM trainer.",
                UserWarning,
            )

        if not return_dict:
            return (logits,) + (outputs[1:] if len(outputs) > 1 else tuple())

        return MetaStateOutput(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
            final_state_rel_norm=diagnostics.get("final_state_rel_norm"),
        )

    def can_generate(self) -> bool:
        return True
