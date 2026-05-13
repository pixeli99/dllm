"""
LLaDA-WoDD modeling.

Subclasses LLaDAModel / LLaDAModelLM so a vanilla LLaDA-8B checkpoint loads
in by-name without conversion. The only structural change is that the
recurrent middle block is run T_step times per forward, and after each pass
the model writes a (soft-gated) latent vector to an append-only memory tape
which subsequent passes can read via TapeRead cross-attention.

INVARIANTS:
  - With zero_init_wodd=True (default), step 0 is bit-identical to vanilla
    split LLaDA: WriteHead.content_proj=0 -> v_k=0 -> tape entries are 0 ->
    TapeRead receives 0-valued K/V -> output is 0 (also because out_proj=0)
    -> h flows through prelude / recurrent / coda exactly as vanilla.
  - Append-only: MemoryTape only supports .append. No overwrite anywhere.
  - The recurrent block weights are SHARED across all T_step iterations.

Memory: full BPTT through T_step iterations multiplies activation memory
~T_step x over a single forward. Use gradient_checkpointing if you OOM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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

from ._wodd_modules import (
    MemoryTape,
    TapeRead,
    WoDDStepStats,
    WriteHead,
    masked_mean,
)
from .configuration_llada_wodd import LLaDAWoDDConfig


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class WoDDLLaDAOutput(CausalLMOutputWithPast):
    """CausalLMOutputWithPast + tape diagnostics for sparsity / entropy loss."""

    write_gates: Optional[torch.Tensor] = None      # (B, T_step, 1)
    tape_diagnostics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LLaDAWoDDModel(LLaDAModel):
    """LLaDAModel with WoDD append-only memory tape.

    Layer partition:
        prelude:    blocks[0:P]
        recurrent:  blocks[P:P+R]    (iterated T_step times, weight-shared)
        coda:       blocks[P+R:P+R+C]

    Each inner iteration k = 1..T_step:
        if tape non-empty: h <- h + TapeRead(h, tape)
        h <- recurrent_blocks(h)
        pool = masked_mean(h)
        g_k, v_k = WriteHead(pool)
        tape.append(g_k * v_k, gate=g_k)
    """

    config_class = LLaDAWoDDConfig

    def __init__(self, config: LLaDAWoDDConfig, init_params: bool = True):
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
        assert int(config.prelude_layers) >= 0
        assert int(config.coda_layers) >= 1

        d = config.d_model
        dev = config.init_device

        # WoDD modules
        self.tape_read = TapeRead(
            d_model=d,
            tape_dim=config.tape_dim,
            max_writes=config.tape_max_writes,
            n_heads=config.tape_n_heads,
            zero_init=config.zero_init_wodd,
            init_device=dev,
        )
        self.write_head = WriteHead(
            d_model=d,
            tape_dim=config.tape_dim,
            zero_init=config.zero_init_wodd,
            init_device=dev,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        output_tape_diagnostics: bool = False,
    ) -> Tuple[LLaDAOutput, Dict[str, Any]]:
        assert not self.config.alibi, "ALiBi not supported for MDM."
        assert self.config.rope, "RoPE required for MDM."
        assert past_key_values is None and not use_cache, "KV cache not supported."
        assert self.config.block_group_size == 1, (
            "LLaDAWoDDModel requires block_group_size=1 "
            f"(got {self.config.block_group_size})."
        )

        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False

        if T_step is None:
            T_step = int(self.config.t_step_eval)
        T_step = max(1, int(T_step))
        if T_step > int(self.config.tape_max_writes):
            raise ValueError(
                f"T_step={T_step} exceeds tape_max_writes="
                f"{self.config.tape_max_writes}"
            )

        # ------------- Embeddings -------------
        if input_embeddings is not None:
            B_, T_, _ = input_embeddings.size()
            x = input_embeddings
        else:
            B_, T_ = input_ids.size()
            x = self.transformer.wte(input_ids)

        if self.config.input_emb_norm:
            x = x * (self.config.d_model ** 0.5)
        x = self.transformer.emb_drop(x)

        # ------------- Attention bias -------------
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

        prelude_end = int(self.config.prelude_layers)
        coda_start = prelude_end + int(self.config.recurrent_layers)
        blocks = self.transformer.blocks
        all_hidden_states: List[torch.Tensor] = []

        # ------------- Prelude -------------
        for block in blocks[:prelude_end]:
            if output_hidden_states:
                all_hidden_states.append(x)
            x, _ = block(x, attention_bias=eff_bias, layer_past=None, use_cache=False)

        # ------------- WoDD inner loop -------------
        tape = MemoryTape(
            batch_size=B_,
            tape_dim=int(self.config.tape_dim),
            max_writes=int(self.config.tape_max_writes),
        )

        h = x
        per_iter_tape_norms: List[torch.Tensor] = []

        for k in range(T_step):
            # 1. Read from tape (if any writes have happened)
            if not tape.is_empty():
                tape_stack = tape.stack()                          # (B, K, tape_dim)
                tape_contrib = self.tape_read(
                    h, tape_stack, attention_mask=attention_mask
                )
                h = h + tape_contrib
                if output_tape_diagnostics:
                    with torch.no_grad():
                        per_iter_tape_norms.append(
                            tape_contrib.norm() / h.norm().clamp_min(1e-6)
                        )

            # 2. Run recurrent blocks (weight-shared across k)
            for block in blocks[prelude_end:coda_start]:
                h, _ = block(
                    h, attention_bias=eff_bias, layer_past=None, use_cache=False
                )

            # 3. Pool and decide write
            pool = masked_mean(h, attention_mask)                 # (B, d_model)
            g_k, v_k = self.write_head(pool)                      # (B,1), (B,d_tape)
            tape.append(g_k * v_k, gate=g_k)

        # ------------- Coda -------------
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

        write_gates = tape.gates()  # (B, T_step, 1)
        diagnostics = None
        if output_tape_diagnostics:
            diagnostics = {
                "tape_contrib_rel_norm": per_iter_tape_norms,  # list of scalars
                "T_step": T_step,
            }

        return out, {
            "write_gates": write_gates,
            "tape_diagnostics": diagnostics,
        }


# ---------------------------------------------------------------------------
# HF wrapper
# ---------------------------------------------------------------------------

class LLaDAWoDDModelLM(LLaDAModelLM):
    config_class = LLaDAWoDDConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDALlamaBlock"]

    def __init__(
        self,
        config: LLaDAWoDDConfig,
        model: Optional[LLaDAWoDDModel] = None,
        init_params: bool = False,
    ):
        LLaDAPreTrainedModel.__init__(self, config)
        if model is None:
            config.init_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = LLaDAWoDDModel(config, init_params=init_params)
        else:
            self.model = model

    @classmethod
    def from_llada_checkpoint(
        cls,
        model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        **wodd_kwargs,
    ) -> "LLaDAWoDDModelLM":
        """Retrofit a pretrained vanilla LLaDA into a WoDD variant.

        Loads vanilla LLaDA weights and copies them into a freshly built
        LLaDAWoDDModelLM. WoDD-specific params (write_head.*, tape_read.*) are
        initialized in their __init__ -- with `zero_init_wodd=True` the
        forward at step 0 is bit-identical to vanilla split LLaDA.
        """
        from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
        from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        base_config = LLaDAConfig.from_pretrained(model_name_or_path)
        config_dict = base_config.to_dict()
        config_dict.pop("model_type", None)
        config_dict.pop("architectures", None)
        config_dict.update(wodd_kwargs)
        wodd_config = cls.config_class(**config_dict)

        vanilla = LLaDAModelLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        )

        model = cls(wodd_config, init_params=False)
        model = model.to(dtype=torch_dtype)

        missing, unexpected = model.load_state_dict(
            vanilla.state_dict(), strict=False
        )
        if unexpected:
            import warnings
            warnings.warn(
                f"Unexpected keys when loading vanilla LLaDA into WoDD: "
                f"{list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}",
                UserWarning,
            )
        # `missing` keys are the WoDD-specific params; expected.

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
        output_tape_diagnostics: bool = False,
    ) -> Union[Tuple, WoDDLLaDAOutput]:
        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not supported in LLaDAWoDD.")
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs, wodd_extras = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            T_step=T_step,
            output_tape_diagnostics=output_tape_diagnostics,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        if labels is not None:
            import warnings
            warnings.warn(
                "LLaDAWoDD does not compute loss inside forward; use MDLMWoDDTrainer.",
                UserWarning,
            )

        if not return_dict:
            return (logits,) + (outputs[1:] if len(outputs) > 1 else tuple())

        return WoDDLLaDAOutput(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
            write_gates=wodd_extras["write_gates"],
            tape_diagnostics=wodd_extras["tape_diagnostics"],
        )

    def can_generate(self) -> bool:
        return True
