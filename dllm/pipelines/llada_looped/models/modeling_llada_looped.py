"""
LLaDA-Looped modeling.

Subclasses LLaDAModel and LLaDAModelLM so a vanilla LLaDA-8B checkpoint
loads in by-name without conversion. The only structural change is the
`forward` method: transformer blocks are partitioned into prelude /
recurrent / coda segments, and the recurrent segment is looped `T_rec`
times with a Parcae-style pre-filter injection:

    h_{r+1} = R(A_disc * h_r + B_disc @ e)

where
    A_cont = -exp(log_A)                   # diagonal, negative
    A_disc = exp(delta * A_cont)           # diagonal, in (0, 1); ZOH
    B_disc = delta * B_inject              # Euler (row-scaled)
    e      = input_norm(prelude_output)

At init (log_A = A_init_log, delta = delta_init, B_inject = 0):
    A_disc ≈ 1, B_disc = 0 -> h_{r+1} ≈ R(h_r)

which with `T_rec = 1` is numerically equivalent to vanilla LLaDA
(minus a near-identity input LayerNorm). This makes retrofit a
drop-in weight load followed by SFT.

Truncated BPTT: the first `T_rec - T_bwd` recurrences run under
`torch.no_grad()`; only the trailing `T_bwd` iterations contribute to
autograd.

Run:
    # This file is a modeling module — no entry point. See:
    #   /Users/pixeli/dllm/examples/llada_looped/sft.py
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from dllm.pipelines.llada.models.modeling_llada import (
    LayerNorm,
    LLaDAModel,
    LLaDAModelLM,
    LLaDAOutput,
    LLaDAPreTrainedModel,
    ensure_finite_,
)

from .configuration_llada_looped import LLaDALoopedConfig


class LLaDALoopedModel(LLaDAModel):
    """LLaDAModel with a middle-loop over `blocks[prelude:coda_start]`."""

    config_class = LLaDALoopedConfig

    def __init__(self, config: LLaDALoopedConfig, init_params: bool = True):
        super().__init__(config, init_params=init_params)
        d = config.d_model

        # --- Parcae stabilizer params ---
        self.log_A = nn.Parameter(torch.full((d,), float(config.A_init_log)))
        self.delta = nn.Parameter(torch.full((d,), float(config.delta_init)))
        if config.use_diag_B:
            self.B_inject = nn.Parameter(
                torch.full((d,), float(config.B_init_scale))
            )
        else:
            if config.B_init_scale == 0.0:
                B_mat = torch.zeros(d, d)
            else:
                # Small random init scaled by B_init_scale / sqrt(d)
                B_mat = torch.randn(d, d) * (config.B_init_scale / math.sqrt(d))
            self.B_inject = nn.Parameter(B_mat)

        # --- Input-injection normalization (LN of prelude output) ---
        if config.use_input_norm:
            self.input_norm = LayerNorm.build(config)
        else:
            self.input_norm = nn.Identity()

    # ---- helpers ----
    def _discrete_AB(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (A_disc, B_disc) via ZOH (for A) / Euler (for B)."""
        A_cont = -torch.exp(self.log_A)                 # [d], negative
        A_disc = torch.exp(self.delta * A_cont)         # [d], in (0, 1)
        if self.config.use_diag_B:
            B_disc = self.delta * self.B_inject         # [d]
        else:
            # Row-scale the matrix by delta (per output-dim).
            B_disc = self.delta.unsqueeze(-1) * self.B_inject  # [d, d]
        return A_disc, B_disc

    def _inject(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        A_disc: torch.Tensor,
        B_disc: torch.Tensor,
    ) -> torch.Tensor:
        """Return A_disc * h + B_disc @ e, shape [B, T, D]."""
        h_filt = h * A_disc  # broadcast [d] over [B, T, D]
        if self.config.use_diag_B:
            e_inj = e * B_disc
        else:
            e_inj = F.linear(e, B_disc)  # [B, T, D] @ [D, D]^T
        return h_filt + e_inj

    # ---- forward ----
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
        T_rec: Optional[int] = None,
        T_bwd: Optional[int] = None,
    ) -> LLaDAOutput:
        assert not self.config.alibi, "ALiBi not supported for MDM."
        assert self.config.rope, "RoPE required for MDM."
        assert past_key_values is None and not use_cache, "KV cache not supported."
        assert self.config.block_group_size == 1, (
            "LLaDALoopedModel requires block_group_size=1 (got "
            f"{self.config.block_group_size})."
        )

        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False

        if T_rec is None:
            T_rec = int(self.config.mu_rec_eval)
        T_rec = max(1, int(T_rec))
        if T_bwd is None:
            T_bwd = T_rec
        T_bwd = max(0, min(int(T_bwd), T_rec))

        if input_embeddings is not None:
            B_, T_, _ = input_embeddings.size()
            x = input_embeddings
        else:
            B_, T_ = input_ids.size()
            x = self.transformer.wte(input_ids)

        if self.config.input_emb_norm:
            x = x * (self.config.d_model ** 0.5)
        x = self.transformer.emb_drop(x)

        # --- Attention bias (bidirectional for MDM + masked padding) ---
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

        # --- Prelude ---
        for block in blocks[:prelude_end]:
            if output_hidden_states:
                all_hidden_states.append(x)
            x, _ = block(x, attention_bias=eff_bias, layer_past=None, use_cache=False)

        # --- Input injection signal (computed once, reused each recurrence) ---
        e = self.input_norm(x)

        # --- Recurrent loop ---
        A_disc, B_disc = self._discrete_AB()
        h = x
        n_no_grad = T_rec - T_bwd
        for r in range(T_rec):
            ctx = torch.no_grad() if r < n_no_grad else nullcontext()
            with ctx:
                h_in = self._inject(h, e, A_disc, B_disc)
                for block in blocks[prelude_end:coda_start]:
                    h_in, _ = block(
                        h_in, attention_bias=eff_bias, layer_past=None, use_cache=False
                    )
                h = h_in

        # --- Coda ---
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

        return LLaDAOutput(
            logits=logits,
            attn_key_values=None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


class LLaDALoopedModelLM(LLaDAModelLM):
    """HF wrapper. Inherits tying / input-embed utilities from LLaDAModelLM."""

    config_class = LLaDALoopedConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDALlamaBlock"]

    def __init__(
        self,
        config: LLaDALoopedConfig,
        model: Optional[LLaDALoopedModel] = None,
        init_params: bool = False,
    ):
        # Skip LLaDAModelLM.__init__ to avoid constructing a throwaway LLaDAModel.
        LLaDAPreTrainedModel.__init__(self, config)
        if model is None:
            self.model = LLaDALoopedModel(config, init_params=init_params)
        else:
            self.model = model

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
        T_rec: Optional[int] = None,
        T_bwd: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not supported in LLaDALooped.")
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
            T_rec=T_rec,
            T_bwd=T_bwd,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states
        loss = None
        if labels is not None:
            import warnings
            warnings.warn(
                "LLaDALooped does not compute loss in forward; use MDLMLoopedTrainer.",
                UserWarning,
            )

        if not return_dict:
            output = (logits,) + (outputs[1:] if isinstance(outputs, tuple) else ())
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    @classmethod
    def from_llada_checkpoint(cls, pretrained_path: str, **loop_kwargs):
        """Build a LLaDALoopedModelLM from a vanilla LLaDA-8B checkpoint.

        All pretrained block weights load in by name. The new loop params
        (`log_A`, `delta`, `B_inject`, `input_norm.*`) are freshly initialized
        from `loop_kwargs` (or defaults), so the resulting model at `T_rec=1`
        is numerically close to vanilla LLaDA.

        Example:
            model = LLaDALoopedModelLM.from_llada_checkpoint(
                "GSAI-ML/LLaDA-8B-Base",
                prelude_layers=8, recurrent_layers=16, coda_layers=8,
                mu_rec_eval=2,
            )
        """
        from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig

        base_cfg = LLaDAConfig.from_pretrained(pretrained_path)
        merged = base_cfg.to_dict()
        merged.update(loop_kwargs)
        # Let LLaDALoopedConfig's class attribute drive model_type / architectures.
        merged.pop("model_type", None)
        merged.pop("architectures", None)
        looped_cfg = cls.config_class(**merged)
        return cls.from_pretrained(pretrained_path, config=looped_cfg)
