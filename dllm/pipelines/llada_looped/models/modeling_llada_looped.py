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


# ----- First-N-forwards numerical diagnostic -----
# Prints min / max / absmax / mean of each stage's tensor on rank 0 for the
# first few forward passes, and raises immediately when a non-finite value
# appears. Set DLLM_LOOP_DIAG_STEPS=0 to disable, or =10 for longer trace.
import os as _os

_DIAG_MAX = int(_os.environ.get("DLLM_LOOP_DIAG_STEPS", "3"))
_DIAG_COUNTER = {"n": 0}


def _is_main_rank() -> bool:
    return _os.environ.get("LOCAL_RANK", "0") == "0"


def _diag(t: torch.Tensor, tag: str) -> None:
    has_nan = bool(torch.isnan(t).any().item())
    has_inf = bool(torch.isinf(t).any().item())
    if has_nan or has_inf:
        n_nan = int(torch.isnan(t).sum().item())
        n_inf = int(torch.isinf(t).sum().item())
        raise RuntimeError(
            f"[loop-diag] NON-FINITE at '{tag}': "
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"nan={n_nan} inf={n_inf}"
        )
    if _DIAG_COUNTER["n"] >= _DIAG_MAX or not _is_main_rank():
        return
    tf = t.detach().float()
    print(
        f"[loop-diag step={_DIAG_COUNTER['n']}] {tag}: "
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={tf.min().item():+.3e} "
        f"max={tf.max().item():+.3e} "
        f"absmax={tf.abs().max().item():.3e} "
        f"mean={tf.mean().item():+.3e} "
        f"std={tf.std().item():.3e}",
        flush=True,
    )


class LLaDALoopedModel(LLaDAModel):
    """LLaDAModel with a middle-loop over `blocks[prelude:coda_start]`."""

    config_class = LLaDALoopedConfig

    def __init__(self, config: LLaDALoopedConfig, init_params: bool = True):
        super().__init__(config, init_params=init_params)
        d = config.d_model
        dev = config.init_device  # e.g. "cuda" — set by LLaDALoopedModelLM.__init__

        # --- Parcae stabilizer params ---
        # A_init_log = 5 -> A_disc = exp(-exp(5)) ≈ 0; with h_0 = 0 this is
        # irrelevant at step 0, but keeps state decayed so new e inputs aren't
        # blown up by a lingering large A·h feedback term in later steps.
        self.log_A = nn.Parameter(
            torch.full((d,), float(config.A_init_log), device=dev)
        )
        self.delta = nn.Parameter(
            torch.full((d,), float(config.delta_init), device=dev)
        )

        # B_inject init = identity so B·e == e at step 0.
        # Combined with h_0 = 0 and e = x_prelude, this gives
        #     h_1 = R(0 + I·x_prelude) = R(x_prelude) = vanilla LLaDA forward.
        if config.use_diag_B:
            if config.B_init_identity:
                self.B_inject = nn.Parameter(torch.ones(d, device=dev))
            else:
                self.B_inject = nn.Parameter(torch.zeros(d, device=dev))
        else:
            if config.B_init_identity:
                self.B_inject = nn.Parameter(torch.eye(d, device=dev))
            else:
                self.B_inject = nn.Parameter(torch.zeros(d, d, device=dev))

        # --- Input-injection normalization (LN of prelude output) ---
        # Default off: e = x_prelude verbatim. Turning this on inserts an
        # extra learnable LN between the prelude and the loop; useful if
        # prelude-output magnitudes drift but breaks the drop-in retrofit.
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
        _diag(x, "0_after_embed")

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
        _diag(x, f"1_after_prelude[{prelude_end}]")

        # --- Input injection signal (computed once, reused each recurrence) ---
        e = self.input_norm(x)
        _diag(e, "2_e=input_norm(x)")

        # --- Recurrent loop ---
        A_disc, B_disc = self._discrete_AB()
        _diag(A_disc, "3_A_disc")
        _diag(B_disc, "3_B_disc")

        h = torch.zeros_like(x)
        _diag(h, "4_h_init=zeros")

        n_no_grad = T_rec - T_bwd
        for r in range(T_rec):
            ctx = torch.no_grad() if r < n_no_grad else nullcontext()
            with ctx:
                h_in = self._inject(h, e, A_disc, B_disc)
                _diag(h_in, f"5_loop[r={r}]/after_inject")
                for bi, block in enumerate(blocks[prelude_end:coda_start]):
                    h_in, _ = block(
                        h_in, attention_bias=eff_bias, layer_past=None, use_cache=False
                    )
                    if bi in (0, len(blocks[prelude_end:coda_start]) - 1):
                        _diag(h_in, f"6_loop[r={r}]/block[{bi}]")
                h = h_in

        # --- Coda ---
        x = h
        for block in blocks[coda_start:]:
            if output_hidden_states:
                all_hidden_states.append(x)
            x, _ = block(x, attention_bias=eff_bias, layer_past=None, use_cache=False)
        _diag(x, "7_after_coda")

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
        _diag(logits, "8_logits")
        _DIAG_COUNTER["n"] += 1

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
            # Mirror vanilla LLaDAModelLM: allocate params directly on GPU so
            # we don't hold an 8B-param CPU-resident copy before FSDP shards.
            config.init_device = "cuda"
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

        model = cls.from_pretrained(pretrained_path, config=looped_cfg)
        # from_pretrained under FSDP uses low_cpu_mem_usage=True, which realizes
        # missing-key parameters as zeros instead of running our __init__ code.
        # Re-seed the loop params here (rank 0 only under cpu_ram_efficient_loading;
        # FSDP sync_module_states=true will broadcast to the rest).
        cls._reset_loop_parameters(model, looped_cfg)
        return model

    @staticmethod
    def _reset_loop_parameters(model: "LLaDALoopedModelLM", config: LLaDALoopedConfig) -> None:
        """Fill log_A / delta / B_inject (and input_norm if used) with their
        intended init values. Safe to call after from_pretrained; idempotent."""
        core = model.model  # LLaDALoopedModel
        is_main = _os.environ.get("LOCAL_RANK", "0") == "0"

        def _fill(p: torch.Tensor, v: float) -> None:
            if p.device.type == "meta":
                return  # Non-rank-0 under cpu_ram_efficient_loading; FSDP broadcasts later.
            with torch.no_grad():
                p.fill_(v)

        if is_main:
            print(
                f"[loop-init] resetting loop params: "
                f"log_A={config.A_init_log} delta={config.delta_init} "
                f"B_init_identity={config.B_init_identity} "
                f"use_diag_B={config.use_diag_B} "
                f"use_input_norm={config.use_input_norm}",
                flush=True,
            )

        _fill(core.log_A, float(config.A_init_log))
        _fill(core.delta, float(config.delta_init))

        B = core.B_inject
        if B.device.type != "meta":
            with torch.no_grad():
                if config.use_diag_B:
                    B.fill_(1.0 if config.B_init_identity else 0.0)
                else:
                    B.zero_()
                    if config.B_init_identity:
                        B.fill_diagonal_(1.0)

        # input_norm's LayerNorm weight/bias also get realized as zeros; restore
        # weight=1, bias=0 so the LN acts as a proper (γ=1, β=0) normalizer.
        if config.use_input_norm and not isinstance(core.input_norm, nn.Identity):
            ln = core.input_norm
            w = getattr(ln, "weight", None)
            b = getattr(ln, "bias", None)
            if w is not None:
                _fill(w, 1.0)
            if b is not None:
                _fill(b, 0.0)
