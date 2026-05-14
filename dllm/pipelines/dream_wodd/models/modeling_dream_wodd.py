"""Dream-WoDD modeling (Phase 5 cross-family).

Partitions Dream-7B's `self.layers` into prelude / recurrent / coda and
runs the recurrent block T_step times per forward, with an append-only
latent tape between iterations. Tape mechanics (WriteHead, TapeRead,
MemoryTape, masked_mean) are imported VERBATIM from the LLaDA-WoDD
pipeline -- they only depend on d_model and tape dims, not the backbone
family.

Drop-in invariant (mirrors LLaDA-WoDD): with `zero_init_wodd=True` and
T_step=1, the forward must be byte-identical to a vanilla DreamModel
forward. Phase 5 sanity should re-run a Phase-1a-style logit-identity
check before any training.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

from dllm.pipelines.dream.models.modeling_dream import (
    DreamBaseModel,
    DreamModel,
    DreamGenerationMixin,
    DreamPreTrainedModel,
)

# Pipeline-agnostic WoDD modules — reused from the LLaDA pipeline.
from dllm.pipelines.llada_wodd.models._wodd_modules import (
    MemoryTape, TapeRead, WriteHead, masked_mean,
)

from .configuration_dream_wodd import DreamWoDDConfig


class DreamWoDDBaseModel(DreamBaseModel):
    """DreamBaseModel with WoDD partition.

    Layer partition (over `self.layers`):
        prelude:    [0 : P]
        recurrent:  [P : P + R]    (re-applied T_step times, weight-shared)
        coda:       [P + R : ...]
    """
    config_class = DreamWoDDConfig

    def __init__(self, config: DreamWoDDConfig):
        super().__init__(config)
        total = (int(config.prelude_layers)
                 + int(config.recurrent_layers)
                 + int(config.coda_layers))
        assert total <= config.num_hidden_layers, (
            f"prelude({config.prelude_layers}) + recurrent({config.recurrent_layers}) "
            f"+ coda({config.coda_layers}) = {total} > num_hidden_layers "
            f"({config.num_hidden_layers})"
        )
        assert int(config.recurrent_layers) >= 1
        assert int(config.coda_layers)      >= 1

        d = int(config.hidden_size)
        self.tape_read = TapeRead(
            d_model=d,
            tape_dim=config.tape_dim,
            max_writes=config.tape_max_writes,
            n_heads=config.tape_n_heads,
            zero_init=config.zero_init_wodd,
            init_device=None,  # Dream's HF post_init handles device
        )
        self.write_head = WriteHead(
            d_model=d,
            tape_dim=config.tape_dim,
            zero_init=config.zero_init_wodd,
            init_device=None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        T_step: Optional[int] = None,
        output_tape_diagnostics: bool = False,
    ) -> BaseModelOutput:
        # Inputs prep (lifted from DreamBaseModel.forward verbatim) ---------
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        from transformers.cache_utils import DynamicCache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # WoDD partition --------------------------------------------------
        if T_step is None:
            T_step = int(self.config.t_step_eval)
        T_step = max(1, int(T_step))
        if T_step > int(self.config.tape_max_writes):
            raise ValueError(
                f"T_step={T_step} exceeds tape_max_writes={self.config.tape_max_writes}"
            )
        P = int(self.config.prelude_layers)
        R = int(self.config.recurrent_layers)
        coda_start = P + R

        # Tape buffer (re-init each forward).
        # For attention_mask in masked_mean we want a (B, L) view; Dream may
        # be passed full 4-D mask. We coerce safely.
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            am_2d = attention_mask
        else:
            am_2d = None

        def run_layer(layer, hs):
            if self.gradient_checkpointing and self.training:
                return self._gradient_checkpointing_func(
                    layer.__call__,
                    hs, attention_mask, position_ids, past_key_values,
                    output_attentions, use_cache, cache_position,
                    position_embeddings,
                )
            return layer(
                hs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        # Prelude
        for layer in self.layers[:P]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            out = run_layer(layer, hidden_states)
            hidden_states = out[0]

        B = hidden_states.shape[0]
        tape = MemoryTape(
            batch_size=B,
            tape_dim=int(self.config.tape_dim),
            max_writes=int(self.config.tape_max_writes),
        )

        per_iter_tape_norms: List[torch.Tensor] = []
        h = hidden_states
        for k in range(T_step):
            if not tape.is_empty():
                contrib = self.tape_read(h, tape.stack(), attention_mask=am_2d)
                if output_tape_diagnostics:
                    with torch.no_grad():
                        per_iter_tape_norms.append(
                            contrib.norm() / h.norm().clamp_min(1e-6)
                        )
                h = h + contrib
            for layer in self.layers[P:coda_start]:
                out = run_layer(layer, h)
                h = out[0]
            pool = masked_mean(h, am_2d) if am_2d is not None else h.mean(dim=1)
            g, v = self.write_head(pool)
            tape.append(g * v, gate=g)

        # Coda
        hidden_states = h
        for layer in self.layers[coda_start:]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            out = run_layer(layer, hidden_states)
            hidden_states = out[0]

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        diagnostics = None
        if output_tape_diagnostics:
            diagnostics = {
                "tape_contrib_rel_norm": per_iter_tape_norms,
                "T_step": T_step,
                "write_gates": tape.gates(),
            }
        # We stash diagnostics on a private attr for the LM wrapper to read.
        self._wodd_diag = diagnostics
        self._wodd_gates = tape.gates()

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DreamWoDDModel(DreamGenerationMixin, DreamPreTrainedModel):
    """LM head over DreamWoDDBaseModel; mirrors DreamModel."""
    config_class = DreamWoDDConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DreamWoDDConfig):
        super().__init__(config)
        self.model = DreamWoDDBaseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        T_step: Optional[int] = None,
        output_tape_diagnostics: bool = False,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Match DreamModel.forward attention_mask massaging.
        if (isinstance(attention_mask, str) and attention_mask == "full") or attention_mask is None:
            pass
        elif isinstance(attention_mask, torch.Tensor):
            if not torch.any(attention_mask == 0.0):
                attention_mask = "full"

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            T_step=T_step,
            output_tape_diagnostics=output_tape_diagnostics,
        )
        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]
        out = MaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # Surface tape diagnostics if requested -- read off the base model.
        out.write_gates = getattr(self.model, "_wodd_gates", None)
        out.tape_diagnostics = getattr(self.model, "_wodd_diag", None)
        return out

    @classmethod
    def from_dream_checkpoint(
        cls,
        model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        **wodd_kwargs,
    ) -> "DreamWoDDModel":
        """Retrofit a pretrained Dream-7B into a Dream-WoDD variant.

        Mirrors LLaDAWoDDModelLM.from_llada_checkpoint: load the vanilla
        Dream weights, build a fresh DreamWoDDModel, and copy by-name
        via load_state_dict(..., strict=False). WoDD modules
        (write_head.*, tape_read.*) are zero-initialised inside the
        DreamWoDDBaseModel constructor when zero_init_wodd=True.
        """
        from dllm.pipelines.dream.models.configuration_dream import DreamConfig
        from dllm.pipelines.dream.models.modeling_dream import DreamModel as _DreamLM

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        base_config = DreamConfig.from_pretrained(model_name_or_path)
        config_dict = base_config.to_dict()
        config_dict.pop("model_type", None)
        config_dict.pop("architectures", None)
        config_dict.update(wodd_kwargs)
        wodd_config = cls.config_class(**config_dict)

        vanilla = _DreamLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
        model = cls(wodd_config)
        model = model.to(dtype=torch_dtype)

        missing, unexpected = model.load_state_dict(
            vanilla.state_dict(), strict=False
        )
        if unexpected:
            import warnings
            warnings.warn(
                f"Unexpected keys loading vanilla Dream into DreamWoDD: "
                f"{list(unexpected)[:8]}", UserWarning,
            )
        del vanilla
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
