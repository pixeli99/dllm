"""
LLaDA-Looped modeling (V2.1-a, feedback-bridge variant).

Subclasses LLaDAModel and LLaDAModelLM so a vanilla LLaDA-8B checkpoint
loads in by-name without conversion. Structural change: blocks are
partitioned into prelude / recurrent (R) / coda; R is iterated `T_rec`
times per forward, with two variants selected by config.

KEY ARCHITECTURAL CHOICES (V2.1-a):
  - First-pass bypass: the RecursiveLink is NEVER applied at r=0. It is
    purely a feedback-transition bridge between iter r-1's R output and
    iter r's R input (for r >= 1). This makes T_rec=1 strictly equivalent
    to vanilla split LLaDA, by construction.
  - Full BPTT: gradient flows through ALL iterations end-to-end, no
    detach. This is the "whole-loop co-optimization" regime -- early
    iterations get direct credit assignment from L_final.

  V2 (use_latent_feedback=True):
      h_0 = e = prelude(x)
      h_1 = R(h_0)                              # first pass: vanilla
      for r = 2, ..., T_rec:
          h_r = R(RecursiveLink(h_{r-1}, r-2))  # feedback (no detach)

      RecursiveLink(h, k) = h + gate_k * proj2(GELU(proj1(pre_ln(h))))
      where gate_k = 1 + tanh(iter_gate_raw[k]) ∈ (0, 2)^d is a per-iter,
      per-channel multiplicative gate on the link's residual output, keyed
      on the feedback-iter index k in {0, ..., T_rec-2}. proj2 and
      iter_gate_raw are both zero-initialized so the loop body starts as
      V1 dynamics (h_{r+1} = R(h_r)) and learns r-specific output scales
      from there. The gate is bounded (tanh range), so adapter_update can
      not run away the way an input-side FiLM (γ on pre_ln output) does --
      the previous V2.1-b design hit denoising-loss-overfitting collapse
      around 5k steps because γ had no upper bound. No post_ln: the
      residual identity path is critical for the I + W2 sigma' W1 Jacobian
      structure (RecursiveMAS Theorem 4.1).

  V1 (use_latent_feedback=False), pure h-recurrence ablation:
      h_0 = e
      h_1 = R(h_0)                          # first pass: vanilla
      for r = 2, ..., T_rec:
          h_r = R(h_{r-1})                  # pure recurrence (no detach)

Both variants at T_rec=1 are numerically identical to vanilla LLaDA
restricted to a (prelude, R, coda) partition.

Memory note: full BPTT through T_rec=6 means activation memory grows
~6x over a single forward. Enable HF Trainer's gradient_checkpointing
in the run script if you OOM.

Run:
    # This module is imported by the loaded model. See:
    #   /Users/pixeli/dllm/examples/llada_looped/sft.py
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

from .configuration_llada_looped import LLaDALoopedConfig


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoopedLLaDAOutput(CausalLMOutputWithPast):
    """Adds loop diagnostics on top of HF's CausalLMOutputWithPast."""

    loop_diagnostics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# RecursiveLink: iter-conditioned residual feedback bridge (V2.1-c, gated).
#
# Forward: h + gate_r * proj2(GELU(proj1(pre_ln(h))))
#   where gate_r = 1 + tanh(iter_gate_raw[r]) is a per-channel scale keyed
#   on the feedback-iteration index r in {0, ..., n_iters_max-1}.
#
# Design notes:
#   - No outer LayerNorm: the residual identity path 'h + ...' is what
#     gives the Jacobian I + W2 sigma' W1 structure. Wrapping in LN
#     destroys both forward identity (post_ln(h) != h) and the gradient
#     stability argument (RecursiveMAS Theorem 4.1).
#   - proj2 zero-init -> at init link(h) = h regardless of r.
#   - iter_gate_raw zero-init -> tanh(0) = 0 -> gate = 1 at init -> the
#     gated form is exactly equivalent to the V2.1-a (time-invariant)
#     link. A V2.1-a stage1 ckpt that lacks iter_gate_raw warm-starts to
#     bit-identical behavior.
#   - Bounded gate ∈ (0, 2)^d. Crucially this prevents the runaway that
#     killed V2.1-b (input-side FiLM with unbounded gamma): there, an
#     unbounded multiplicative modulation BEFORE the proj1 / GELU / proj2
#     pipeline let adapter_update scale up to 5x ||h|| and the system
#     overfit MDLM denoising loss while collapsing on GSM8K (-15 pts at
#     5k vs 1k). Output-side gating with tanh saturation can not do that.
#   - Combined with first-pass bypass in LLaDALoopedModel.forward, this
#     guarantees T_rec=1 is exactly vanilla split LLaDA.
#
# Why iter-conditioning: the V2.1-a time-invariant link exhibits a "T=2
# trap" -- evaluation peaks at T_rec=2 and drops at T=3,4. Hypothesis:
# the right scale of the residual feedback differs across iters (iter 0:
# big reshape; iter 1+: small refinement). A per-iter scalar/vector gate
# on the link's *output* gives exactly that, without input-side modulation
# capacity that V2.1-b proved is too easy to overfit. Cost: n_iters_max *
# d_model params (~65K), negligible vs the ~50M base link.
# ---------------------------------------------------------------------------

class RecursiveLink(nn.Module):
    def __init__(
        self,
        d_model: int,
        init_device: Optional[str] = None,
        n_iters_max: int = 16,
    ):
        super().__init__()
        self.n_iters_max = int(n_iters_max)
        self.pre_ln = nn.LayerNorm(d_model, device=init_device)
        self.proj1 = nn.Linear(d_model, d_model, device=init_device)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_model, d_model, device=init_device)

        # Per-iter, per-channel multiplicative gate on the link's residual
        # output. raw param (no nonlinearity stored) -> apply 1 + tanh(.) at
        # forward to bound the effective scale to (0, 2).
        self.iter_gate_raw = nn.Parameter(
            torch.zeros(self.n_iters_max, d_model, device=init_device)
        )

        # Zero-init proj2 -> link contribution is 0 at init (h + 0 = h).
        # Zero-init iter_gate_raw -> 1 + tanh(0) = 1 -> at init the gated form
        # is EXACTLY equivalent to the time-invariant V2.1-a link. A V2.1-a
        # ckpt loaded into this module gives bit-identical behavior at step 0.
        with torch.no_grad():
            self.proj2.weight.zero_()
            if self.proj2.bias is not None:
                self.proj2.bias.zero_()
            # iter_gate_raw is already zero-initialized via torch.zeros above.

    def forward(self, h: torch.Tensor, iter_idx: int = 0) -> torch.Tensor:
        # Saturate at the last bucket if T_rec exceeds n_iters_max -- late
        # iters share a gate (asymptotic regime).
        idx = min(int(iter_idx), self.n_iters_max - 1)
        out = self.proj2(self.act(self.proj1(self.pre_ln(h))))
        gate = 1.0 + torch.tanh(self.iter_gate_raw[idx])  # (d,), ∈ (0, 2)
        return h + gate * out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LLaDALoopedModel(LLaDAModel):
    """LLaDAModel with a latent-feedback loop over middle blocks.

    Layer partition:
        prelude:  blocks[0 : P]
        R:        blocks[P : P+R]    (looped T_rec times)
        coda:     blocks[P+R : P+R+C]

    where P, R, C = config.prelude_layers, .recurrent_layers, .coda_layers.
    """

    config_class = LLaDALoopedConfig

    def __init__(self, config: LLaDALoopedConfig, init_params: bool = True):
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
        assert int(config.recurrent_layers) >= 1, "recurrent_layers must be >= 1"
        assert int(config.prelude_layers) >= 0, "prelude_layers must be >= 0"
        assert int(config.coda_layers) >= 1, "coda_layers must be >= 1"

        d = config.d_model
        dev = config.init_device  # set to "cuda" by LLaDALoopedModelLM.__init__

        if config.use_latent_feedback:
            # RecursiveLink: residual feedback bridge applied on iterations
            # r >= 1 only (first-pass bypass in forward).
            self.recursive_link = RecursiveLink(
                d_model=d, init_device=dev
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
        T_rec: Optional[int] = None,
        output_loop_diagnostics: bool = False,
    ) -> Tuple[LLaDAOutput, Dict[str, Any]]:
        assert not self.config.alibi, "ALiBi not supported for MDM."
        assert self.config.rope, "RoPE required for MDM."
        assert past_key_values is None and not use_cache, "KV cache not supported."
        assert self.config.block_group_size == 1, (
            "LLaDALoopedModel requires block_group_size=1 "
            f"(got {self.config.block_group_size})."
        )

        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False

        if T_rec is None:
            T_rec = int(self.config.mu_rec_eval)
        T_rec = max(1, int(T_rec))

        # ------------- Embeddings & attention bias (vanilla LLaDA path) -------------
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

        prelude_end = int(self.config.prelude_layers)
        coda_start = prelude_end + int(self.config.recurrent_layers)
        blocks = self.transformer.blocks

        all_hidden_states: List[torch.Tensor] = []

        # ------------- Prelude -------------
        for block in blocks[:prelude_end]:
            if output_hidden_states:
                all_hidden_states.append(x)
            x, _ = block(x, attention_bias=eff_bias, layer_past=None, use_cache=False)

        e = x  # prelude output (h_0 input)

        # ------------- Recurrent loop (V2.1-a: first-pass bypass, full BPTT) -------------
        # r == 0 (first pass):
        #   h_input = h        (no link -- vanilla R input)
        # r >= 1 (feedback transitions):
        #   h_input = link(h)  for V2
        #   h_input = h        for V1
        #
        # No detach anywhere -- gradient flows end-to-end through every
        # iteration ("whole-loop co-optimization"). Each iter's R/link
        # gets credit from L_final.
        #
        # The first-pass bypass guarantees T_rec=1 is exactly vanilla split
        # LLaDA. The link is a feedback-transition bridge, not a modifier
        # of prelude output.
        diag_residual: List[torch.Tensor] = []
        diag_adapter_update: List[torch.Tensor] = []  # only logged for r >= 1

        h = e
        prev_h: Optional[torch.Tensor] = None

        for r in range(T_rec):
            if r == 0:
                # First pass: no link, no detach.
                h_input = h
            else:
                # Feedback iteration: full BPTT through link + R.
                # iter_idx = r - 1 so the first feedback transition (r=1) maps
                # to iter_gate_raw[0]. Lets the link's output scale differ per
                # feedback iter (e.g. iter 0 large, iter 2 smaller).
                if self.config.use_latent_feedback:
                    h_input = self.recursive_link(h, iter_idx=r - 1)
                    if output_loop_diagnostics:
                        with torch.no_grad():
                            update = (h_input - h).norm() / h.norm().clamp_min(1e-6)
                            diag_adapter_update.append(update)
                else:
                    # V1: pure h-recurrence, no link.
                    h_input = h

            # Run R blocks (16 layers, weight-shared across iters).
            h_out = h_input
            for block in blocks[prelude_end:coda_start]:
                h_out, _ = block(
                    h_out, attention_bias=eff_bias, layer_past=None, use_cache=False
                )

            # Diagnostics (no_grad). Records ||h_r - h_{r-1}|| / ||h_{r-1}||
            # for r >= 1 (i.e., len = T_rec - 1).
            if output_loop_diagnostics and prev_h is not None:
                with torch.no_grad():
                    rel = (h_out - prev_h).norm() / prev_h.norm().clamp_min(1e-6)
                    diag_residual.append(rel)
            prev_h = h_out

            h = h_out

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
        loop_extras = {
            "loop_diagnostics": (
                {
                    "residual_norm": diag_residual,
                    "adapter_update_norm": diag_adapter_update,
                    "T_rec": T_rec,
                }
                if output_loop_diagnostics
                else None
            ),
        }
        return out, loop_extras


# ---------------------------------------------------------------------------
# HF wrapper
# ---------------------------------------------------------------------------

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
        # Skip LLaDAModelLM.__init__ to avoid constructing a throwaway base
        # LLaDAModel.
        LLaDAPreTrainedModel.__init__(self, config)
        if model is None:
            config.init_device = "cuda"
            self.model = LLaDALoopedModel(config, init_params=init_params)
        else:
            self.model = model

    @classmethod
    def from_llada_checkpoint(
        cls,
        model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        **loop_kwargs,
    ) -> "LLaDALoopedModelLM":
        """Retrofit a pretrained vanilla LLaDA checkpoint into a looped variant.

        Loads vanilla LLaDA-8B weights and copies them into a freshly built
        LLaDALoopedModelLM. Loop-specific parameters (`recursive_link.*`) are
        initialized in the link's __init__: pre_ln / proj1 use PyTorch
        defaults, proj2 weight + bias zero-initialized so link is identity
        at init.

        Combined with first-pass bypass in forward, T_rec=1 is exactly
        vanilla LLaDA -- both at init AND after training (architectural
        drop-in, not just numerical).

        For resumes from a looped checkpoint, use the standard `from_pretrained`
        path -- the state_dict will contain recursive_link weights, and AutoConfig
        resolves to LLaDALoopedConfig via the registered `model_type`.

        Args:
            model_name_or_path: vanilla LLaDA checkpoint dir or HF id.
            torch_dtype: dtype for the materialized model (default bf16).
            **loop_kwargs: forwarded to LLaDALoopedConfig
                (prelude_layers, recurrent_layers, coda_layers,
                 use_latent_feedback, mu_rec_eval).
        """
        from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
        from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        base_config = LLaDAConfig.from_pretrained(model_name_or_path)
        config_dict = base_config.to_dict()
        config_dict.pop("model_type", None)
        config_dict.pop("architectures", None)
        config_dict.update(loop_kwargs)
        looped_config = cls.config_class(**config_dict)

        vanilla = LLaDAModelLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        )

        model = cls(looped_config, init_params=False)
        model = model.to(dtype=torch_dtype)

        missing, unexpected = model.load_state_dict(
            vanilla.state_dict(), strict=False
        )
        if unexpected:
            import warnings
            warnings.warn(
                f"Unexpected keys when loading vanilla LLaDA into looped: "
                f"{list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}",
                UserWarning,
            )
        # `missing` should be exactly the loop-specific params; that's expected.

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
        T_rec: Optional[int] = None,
        output_loop_diagnostics: bool = False,
    ) -> Union[Tuple, LoopedLLaDAOutput]:
        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not supported in LLaDALooped.")
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs, loop_extras = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            T_rec=T_rec,
            output_loop_diagnostics=output_loop_diagnostics,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        if labels is not None:
            import warnings
            warnings.warn(
                "LLaDALooped does not compute loss inside forward; "
                "use MDLMLoopedTrainer.",
                UserWarning,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return LoopedLLaDAOutput(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
            loop_diagnostics=loop_extras["loop_diagnostics"],
        )

    def can_generate(self) -> bool:
        return True
