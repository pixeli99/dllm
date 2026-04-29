"""
LLaDA-Looped modeling (belief-bottleneck loop).

Subclasses LLaDAModel and LLaDAModelLM so a vanilla LLaDA-8B checkpoint
loads in by-name without conversion. Structural change: blocks are
partitioned into prelude / recurrent (R) / coda; R is iterated `T_rec`
times per forward, with two variants selected by config:

  V2 (use_belief_feedback=True), the headline design:
      h_0 = e = prelude(x)
      for r = 1, ..., T_rec:
          logits_{r-1} = lm_head(LN_out(h_{r-1}.detach()))   # mid readout
          belief_{r-1} = softmax(logits_{r-1} / tau)           # tau fixed
          f_{r-1}      = belief_{r-1} @ embed_matrix           # soft re-embed
          f_{r-1}      = belief_norm(f_{r-1})                  # RMS magnitude align
          g_{r-1}      = alpha * mask_indicator * f_{r-1}      # mask-aware gate
          h_r          = R(h_{r-1}.detach() + g_{r-1})         # 1-step truncated BPTT

  V1 (use_belief_feedback=False), pure h-recurrence ablation:
      h_r = R(h_{r-1}.detach())

Drop-in invariant: at alpha=0 and T_rec=1 (V2), or T_rec=1 (V1), the model
is numerically equivalent to vanilla LLaDA forward (a single sweep through
prelude + R + coda blocks).

BPTT: 1-step truncated -- h_in is detached every iteration. R + alpha get
gradient only via L_final through the last iteration. Per-iter mid-readout
logits get gradient to LN_out + lm_head only (no R credit assignment
across iterations). DEQ literature shows this is sound when R becomes
contractive; the trainer monitors KL trajectory as a fixed-point check.

Run:
    # This module is imported by the loaded model. See:
    #   /Users/pixeli/dllm/examples/llada_looped/sft.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoopedLLaDAOutput(CausalLMOutputWithPast):
    """Adds intermediate-iter logits and loop diagnostics on top of HF's
    CausalLMOutputWithPast."""

    intermediate_logits: Optional[List[torch.Tensor]] = None
    loop_diagnostics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# RMSNorm-style magnitude alignment (single-scale, no mean-subtraction).
# Used to align belief_embedding magnitude to residual-stream magnitude
# without introducing a learnable d->d basis rotation.
# ---------------------------------------------------------------------------

class _BeliefNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6, init_device: Optional[str] = None):
        super().__init__()
        self.eps = eps
        # init scale = 1 -> init behavior = "rescale to unit RMS"
        self.scale = nn.Parameter(torch.ones(d_model, device=init_device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        return (x / rms) * self.scale


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LLaDALoopedModel(LLaDAModel):
    """LLaDAModel with a belief-bottleneck loop over middle blocks.

    Layer partition:
        prelude:  blocks[0 : P]
        R:        blocks[P : P+R]    (looped T_rec times)
        coda:     blocks[P+R : P+R+C]

    where P, R, C = config.prelude_layers, .recurrent_layers, .coda_layers.
    """

    config_class = LLaDALoopedConfig

    def __init__(self, config: LLaDALoopedConfig, init_params: bool = True):
        super().__init__(config, init_params=init_params)

        # Partition validity (config skips this so HF to_diff_dict doesn't
        # crash on the GPT-2-scale defaults).
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

        # Belief-feedback parameters live on the model only when V2 is on.
        # For V1 (pure h-recurrence) we add nothing -- just iterate R.
        if config.use_belief_feedback:
            # alpha: scalar gate, init 0 -> drop-in at T_rec=1
            self.alpha = nn.Parameter(
                torch.tensor(float(config.alpha_init), device=dev)
            )

            # LN_out: mid-readout LayerNorm, applied to h_r before lm_head.
            # We initialize from the first coda block's attn_norm (which is
            # the LN that vanilla LLaDA uses to normalize h_{P+R} for
            # processing). See `_init_loop_modules_from_base()` for the copy.
            self.ln_out = LayerNorm.build(config)

            # belief_norm: RMS-magnitude alignment, init scale 1.
            if config.use_belief_norm:
                self.belief_norm = _BeliefNorm(d, init_device=dev)
            else:
                self.belief_norm = nn.Identity()

            # tau: FIXED scalar (not learned). Stored as a non-persistent
            # buffer so it travels with state_dict for resumes but is
            # rebroadcast from config on a clean load. Persistent=False
            # would also work; choose persistent=True so a saved checkpoint
            # remembers exactly which tau was trained with.
            self.register_buffer(
                "tau",
                torch.tensor(float(config.tau), device=dev),
                persistent=True,
            )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_loop_modules_from_base(self) -> None:
        """Copy LN_out weights from the first coda block's attn_norm.

        Called by LLaDALoopedModelLM after the base LLaDA weights have been
        loaded. This gives LN_out a sensible init that matches the
        distribution of h_{P+R} (which is what the first coda block's
        attn_norm was originally designed to normalize).
        """
        if not self.config.use_belief_feedback:
            return

        coda_first_idx = (
            int(self.config.prelude_layers) + int(self.config.recurrent_layers)
        )
        if coda_first_idx >= len(self.transformer.blocks):
            return

        coda_first = self.transformer.blocks[coda_first_idx]
        src_ln = getattr(coda_first, "attn_norm", None)
        if src_ln is None:
            return

        # LayerNorm subclasses may have weight (and bias) or neither.
        if hasattr(src_ln, "weight") and src_ln.weight is not None:
            if hasattr(self.ln_out, "weight") and self.ln_out.weight is not None:
                self.ln_out.weight.copy_(src_ln.weight)
        if hasattr(src_ln, "bias") and src_ln.bias is not None:
            if hasattr(self.ln_out, "bias") and self.ln_out.bias is not None:
                self.ln_out.bias.copy_(src_ln.bias)

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------

    def _embed_weight(self) -> torch.Tensor:
        """Return the embedding matrix used as both input embedding and
        (when weight-tied) lm_head projection. Shape [V, d]."""
        return self.transformer.wte.weight

    def _readout_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Apply LN_out + lm_head to a hidden state to get mid-layer logits.
        Caller is responsible for detaching `h` if a 1-step BPTT boundary
        is desired."""
        x = self.ln_out(h)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits = logits * (1.0 / math.sqrt(self.config.d_model))
        return logits

    def _belief_embedding(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute soft belief embedding `softmax(logits / tau) @ embed`.

        For V=128K and large B*L, the full matmul is the cheaper of two
        equivalent forms: O(B*L*V*d) -> we just call F.linear since
        embed_matrix is [V, d] and softmax is [B, L, V].
        """
        tau = self.tau
        belief = F.softmax(logits / tau, dim=-1)  # [B, L, V]
        # belief @ embed: [B,L,V] @ [V,d] = [B,L,d]
        embed_w = self._embed_weight()  # [V, d]
        f = F.linear(belief, embed_w.t(), None)  # equivalent to belief @ embed_w
        return f

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
        mask_token_id: Optional[int] = None,
        alpha_scale: float = 1.0,
        output_intermediate_logits: bool = False,
        output_loop_diagnostics: bool = False,
    ) -> LLaDAOutput:
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

        e = x  # h_0 = prelude output

        # ------------- Mask indicator (V2 only) -------------
        mask_indicator: Optional[torch.Tensor] = None
        if self.config.use_belief_feedback and input_ids is not None:
            if mask_token_id is None:
                # Caller should pass mask_token_id; if absent, fall back to
                # injecting at all positions (degenerate but safe -- model
                # learns the right gating via alpha and the mask-only
                # inductive bias is lost).
                mask_indicator = torch.ones(B_, T_, 1, device=x.device, dtype=x.dtype)
            else:
                mask_indicator = (input_ids == int(mask_token_id)).to(dtype=x.dtype).unsqueeze(-1)

        # ------------- Recurrent loop -------------
        intermediate_logits: List[torch.Tensor] = []
        diag_entropy: List[torch.Tensor] = []
        diag_residual: List[torch.Tensor] = []
        diag_kl: List[torch.Tensor] = []

        h = e
        prev_inter_logits: Optional[torch.Tensor] = None
        prev_h: Optional[torch.Tensor] = None

        for r in range(T_rec):
            # 1-step truncated BPTT: detach incoming h every iteration.
            # Last iter still has gradient through R because the OUTPUT h
            # (assigned to `h` at end of this iter) is what reaches L_final
            # downstream of the loop.
            h_in = h.detach()

            # Mid-layer readout. Only available when V2 is on (V1 has no
            # ln_out / lm_head loop path), so we silently no-op for V1 even
            # when output_intermediate_logits is requested.
            inter_logits: Optional[torch.Tensor] = None
            if self.config.use_belief_feedback:
                inter_logits = self._readout_logits(h_in)
                if output_intermediate_logits:
                    intermediate_logits.append(inter_logits)

            # Belief-feedback gated injection (V2).
            if self.config.use_belief_feedback:
                f = self._belief_embedding(inter_logits)
                f = self.belief_norm(f)
                effective_alpha = self.alpha * float(alpha_scale)
                if mask_indicator is not None:
                    g = effective_alpha * mask_indicator * f
                else:
                    g = effective_alpha * f
                h_input = h_in + g
            else:
                h_input = h_in

            # Run R blocks (the 16 looped layers, weight-shared across iters).
            h_out = h_input
            for block in blocks[prelude_end:coda_start]:
                h_out, _ = block(
                    h_out, attention_bias=eff_bias, layer_past=None, use_cache=False
                )

            # Diagnostics (no_grad): cheap scalar stats per iter.
            if output_loop_diagnostics:
                with torch.no_grad():
                    if inter_logits is not None:
                        # entropy of belief, mean over batch and positions
                        log_probs = F.log_softmax(inter_logits / self.tau, dim=-1)
                        probs = log_probs.exp()
                        ent = -(probs * log_probs).sum(dim=-1).mean()
                        diag_entropy.append(ent.detach())
                        # KL between successive iterates' beliefs
                        if prev_inter_logits is not None:
                            prev_log_probs = F.log_softmax(prev_inter_logits / self.tau, dim=-1)
                            prev_probs = prev_log_probs.exp()
                            kl = (prev_probs * (prev_log_probs - log_probs)).sum(dim=-1).mean()
                            diag_kl.append(kl.detach())
                        prev_inter_logits = inter_logits
                    if prev_h is not None:
                        # ||h_r - h_{r-1}|| / ||h_{r-1}||
                        rel = (h_out - prev_h).norm() / prev_h.norm().clamp_min(1e-6)
                        diag_residual.append(rel.detach())
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
        # Stash extras as attributes (NamedTuple is immutable, so we wrap).
        # The LM wrapper repackages these into a LoopedLLaDAOutput.
        loop_extras = {
            "intermediate_logits": intermediate_logits if output_intermediate_logits else None,
            "loop_diagnostics": (
                {
                    "entropy": diag_entropy,
                    "residual_norm": diag_residual,
                    "kl_successive": diag_kl,
                    "alpha": self.alpha.detach() if self.config.use_belief_feedback else None,
                    "tau": self.tau.detach() if self.config.use_belief_feedback else None,
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
        LLaDALoopedModelLM. Loop-specific parameters (`alpha`, `ln_out.*`,
        `belief_norm.*`) are initialized fresh; `ln_out` is then re-init'd
        by copying from coda[0].attn_norm via `_init_loop_modules_from_base`.

        Use this for V1 / V2 retrofit. For resumes from a looped checkpoint,
        use the standard `from_pretrained` path -- the state_dict will
        contain `ln_out` and friends and the auto-registered AutoConfig
        resolves to LLaDALoopedConfig so the layer split is preserved.

        Args:
            model_name_or_path: vanilla LLaDA checkpoint dir or HF id.
            torch_dtype: dtype for the materialized model (default bf16).
            **loop_kwargs: forwarded to LLaDALoopedConfig
                (prelude_layers, recurrent_layers, coda_layers,
                 use_belief_feedback, alpha_init, tau, belief_top_k,
                 use_belief_norm, mu_rec_eval).
        """
        from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
        from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        base_config = LLaDAConfig.from_pretrained(model_name_or_path)
        config_dict = base_config.to_dict()
        # Strip identifiers that should be reset for the looped class.
        config_dict.pop("model_type", None)
        config_dict.pop("architectures", None)
        config_dict.update(loop_kwargs)
        looped_config = cls.config_class(**config_dict)

        vanilla = LLaDAModelLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        )

        model = cls(looped_config, init_params=False)
        # Move to vanilla's dtype to keep state_dict load straightforward.
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

        # Initialize LN_out from coda[0].attn_norm now that base weights are
        # loaded. Idempotent guard for accidental re-call.
        model.model._init_loop_modules_from_base()

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
        mask_token_id: Optional[int] = None,
        alpha_scale: float = 1.0,
        output_intermediate_logits: bool = False,
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
            mask_token_id=mask_token_id,
            alpha_scale=alpha_scale,
            output_intermediate_logits=output_intermediate_logits,
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
            intermediate_logits=loop_extras["intermediate_logits"],
            loop_diagnostics=loop_extras["loop_diagnostics"],
        )

    def can_generate(self) -> bool:
        return True
