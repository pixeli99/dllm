"""LLaDA-DPad modeling.

Single forward over [seq | scratchpad]. The model sees an effective
sequence length of T + K, runs the 32 LLaDA blocks once, then strips the
scratchpad before producing logits at the original positions.

Invariant (weaker than WoDD/MetaState):
  with `zero_init_dpad=True`, scratchpad embeddings start at 0. K and V
  at scratch positions are therefore 0 (LayerNorm preserves zero, the
  qkv projections have no bias in LLaDA-8B-Base). HOWEVER, attaching
  K extra positions to the input changes the softmax denominator at
  every original position, so even at init the original-position
  logits drift from vanilla LLaDA by O(K/T). Empirically ~0.3 max-abs
  drift at toy scale (T=8, K=8). This is *expected* and *structural*
  for DPad; training is supposed to absorb the offset. The strict
  drop-in invariant only holds for WoDD and MetaState.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

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

from .configuration_llada_dpad import LLaDADPadConfig


class LLaDADPadModel(LLaDAModel):
    config_class = LLaDADPadConfig

    def __init__(self, config: LLaDADPadConfig, init_params: bool = True):
        super().__init__(config, init_params=init_params)

        d = int(config.d_model)
        K = int(config.dpad_len)
        dev = config.init_device

        # Learnable scratchpad embeddings: one vector per scratch position.
        init = torch.zeros(K, d, device=dev)
        if not config.zero_init_dpad:
            init.normal_(mean=0.0, std=0.02)
        # nn.Parameter rather than Embedding because we want it to behave
        # like a static, fully-batched (K, d) bank with no lookup index.
        self.scratchpad = nn.Parameter(init)

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
        T_step: Optional[int] = None,  # accepted for cross-arm uniformity; ignored.
    ) -> LLaDAOutput:
        assert not self.config.alibi
        assert self.config.rope
        assert past_key_values is None and not use_cache
        assert self.config.block_group_size == 1

        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False

        if input_embeddings is not None:
            B_, T_, _ = input_embeddings.size()
            x = input_embeddings
        else:
            B_, T_ = input_ids.size()
            x = self.transformer.wte(input_ids)

        if self.config.input_emb_norm:
            x = x * (self.config.d_model ** 0.5)
        x = self.transformer.emb_drop(x)

        # Append scratchpad. Concatenated along the token axis: positions
        # T_..T_+K hold the learnable scratch.
        K = int(self.config.dpad_len)
        if K > 0:
            scratch = self.scratchpad.to(dtype=x.dtype, device=x.device)
            scratch = scratch.unsqueeze(0).expand(B_, -1, -1)            # (B, K, d)
            x = torch.cat([x, scratch], dim=1)                            # (B, T+K, d)
            # Extend the attention mask: scratch positions are visible.
            if attention_mask is not None:
                k_mask = torch.ones(B_, K, dtype=attention_mask.dtype,
                                    device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, k_mask], dim=1)
        T_ext = T_ + K

        if attention_mask is not None and 0.0 in attention_mask:
            am = attention_mask.to(dtype=torch.float).view(B_, -1)[:, None, None, :]
            am = (1.0 - am) * torch.finfo(am.dtype).min
        else:
            am = None

        eff_bias: Optional[torch.Tensor] = None
        if attention_bias is not None or am is not None:
            if attention_bias is None:
                eff_bias = self.get_bidirectional_attention_bias(T_ext, x.device)
            else:
                eff_bias = attention_bias
                if eff_bias.dtype in (torch.int8, torch.bool):
                    eff_bias = eff_bias.to(dtype=torch.float)
                    eff_bias.masked_fill_(eff_bias == 0.0, torch.finfo(eff_bias.dtype).min)
            mask_len = T_ext if am is None else am.shape[-1]
            eff_bias = eff_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            if am is not None:
                eff_bias = eff_bias + am
                ensure_finite_(eff_bias, check_neg_inf=True, check_pos_inf=False)

        blocks = self.transformer.blocks
        all_hidden_states: List[torch.Tensor] = []

        # Single forward over [seq | scratchpad] -- T_step is intentionally
        # ignored (DPad is by design a 1-pass arm; see WODD_PLAN.md §3.2).
        for block in blocks:
            if output_hidden_states:
                all_hidden_states.append(x)
            x, _ = block(x, attention_bias=eff_bias, layer_past=None, use_cache=False)

        # Strip scratchpad: emit logits only over the original T_ positions.
        x = x[:, :T_, :]

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

        return LLaDAOutput(
            logits=logits,
            attn_key_values=None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


class LLaDADPadModelLM(LLaDAModelLM):
    config_class = LLaDADPadConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDALlamaBlock"]

    def __init__(
        self,
        config: LLaDADPadConfig,
        model: Optional[LLaDADPadModel] = None,
        init_params: bool = False,
    ):
        LLaDAPreTrainedModel.__init__(self, config)
        if model is None:
            config.init_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = LLaDADPadModel(config, init_params=init_params)
        else:
            self.model = model

    @classmethod
    def from_llada_checkpoint(
        cls,
        model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        **dpad_kwargs,
    ) -> "LLaDADPadModelLM":
        from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
        from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        base_config = LLaDAConfig.from_pretrained(model_name_or_path)
        config_dict = base_config.to_dict()
        config_dict.pop("model_type", None)
        config_dict.pop("architectures", None)
        config_dict.update(dpad_kwargs)
        dpad_config = cls.config_class(**config_dict)

        vanilla = LLaDAModelLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        )

        model = cls(dpad_config, init_params=False)
        model = model.to(dtype=torch_dtype)

        missing, unexpected = model.load_state_dict(
            vanilla.state_dict(), strict=False
        )
        if unexpected:
            import warnings
            warnings.warn(
                f"Unexpected keys loading vanilla LLaDA into DPad: "
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
        T_step: Optional[int] = None,  # accepted for uniformity, ignored.
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not supported in LLaDA-DPad.")
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            T_step=T_step,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        if labels is not None:
            import warnings
            warnings.warn(
                "LLaDA-DPad does not compute loss inside forward; use the MDLM trainer.",
                UserWarning,
            )

        if not return_dict:
            return (logits,) + (outputs[1:] if len(outputs) > 1 else tuple())

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True
