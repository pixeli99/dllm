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
      h_1 = R(h_0)                          # first pass: vanilla
      for r = 2, ..., T_rec:
          proposed = R(RecursiveLink(h_{r-1}))
          h_r = proposed                     # default feedback transition
          # optional:
          # h_r = h_{r-1} + alpha_r * (proposed - h_{r-1})
          # where alpha_r can be constant, normalized by T_rec, or decayed.

      RecursiveLink(h) = h + proj2(GELU(proj1(pre_ln(h))))
      proj2 is zero-initialized so the loop body starts as V1 dynamics
      (h_{r+1} = R(h_r)) and learns a residual perturbation from there.
      No post_ln: the residual identity path is critical for the
      I + W2 sigma' W1 Jacobian structure (RecursiveMAS Theorem 4.1).

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
    # Per-iteration token hidden states after the recurrent loop (length T_rec).
    # Each tensor is [B, L, d_model]; feedback workspace slots are not returned.
    # Only populated when return_loop_hiddens=True is passed to forward.
    # Consumed by deep-supervision auxiliary loss in MDLMLoopedTrainer.
    loop_hiddens: Optional[List[torch.Tensor]] = None


# ---------------------------------------------------------------------------
# RecursiveLink: residual feedback bridge.
#
# Forward: h + proj2(GELU(proj1(pre_ln(h))))
#
# Design notes:
#   - No outer LayerNorm: the residual identity path 'h + ...' is what
#     gives the Jacobian I + W2 sigma' W1 structure. Wrapping in LN
#     destroys both forward identity (post_ln(h) != h) and the gradient
#     stability argument (RecursiveMAS Theorem 4.1).
#   - proj2 weight + bias zero-initialized: at init, link(h) = h, so the
#     loop body degenerates to V1 dynamics h_{r+1} = R(h_r), and link
#     learns a residual perturbation from there.
#   - Combined with first-pass bypass in LLaDALoopedModel.forward, this
#     guarantees T_rec=1 is exactly vanilla split LLaDA.
# ---------------------------------------------------------------------------

class RecursiveLink(nn.Module):
    def __init__(
        self,
        d_model: int,
        init_device: Optional[str] = None,
    ):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d_model, device=init_device)
        self.proj1 = nn.Linear(d_model, d_model, device=init_device)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_model, d_model, device=init_device)

        # Zero-init proj2 -> link is identity at init (h + 0 = h).
        with torch.no_grad():
            self.proj2.weight.zero_()
            if self.proj2.bias is not None:
                self.proj2.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.pre_ln(h)
        out = self.proj2(self.act(self.proj1(x)))
        return h + out


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
        if (
            config.use_damped_update
            and config.damping_schedule == "constant"
            and getattr(config, "learn_damping_alpha", False)
        ):
            alpha = float(config.damping_alpha)
            alpha = min(max(alpha, 1e-6), 1.0 - 1e-6)
            logit = math.log(alpha / (1.0 - alpha))
            self.damping_alpha_logit = nn.Parameter(
                torch.tensor(logit, device=dev)
            )
        if (
            config.use_damped_update
            and config.damping_schedule in {"normalized", "decay"}
            and getattr(config, "learn_damping_tau", True)
        ):
            tau = max(float(config.damping_tau), 1e-6)
            self.damping_tau_logit = nn.Parameter(
                torch.tensor(math.log(math.expm1(tau)), device=dev)
            )

        # Feedback-only workspace slots. They are appended to R inputs only for
        # feedback iterations (r >= 1), never to prelude/coda or the first pass.
        # Always init even when init_params=False: these parameters are absent
        # from vanilla checkpoints.
        workspace_size = int(getattr(config, "workspace_size", 0))
        if workspace_size > 0:
            self.workspace_slots = nn.Parameter(
                torch.empty(workspace_size, d, device=dev)
            )
            nn.init.normal_(
                self.workspace_slots,
                std=float(getattr(config, "workspace_init_std", 0.02)),
            )

    def _damping_tau(self, ref: torch.Tensor) -> torch.Tensor:
        """Return the total feedback refinement budget."""
        dtype = ref.dtype
        device = ref.device
        if bool(getattr(self.config, "learn_damping_tau", True)) and hasattr(
            self, "damping_tau_logit"
        ):
            return F.softplus(self.damping_tau_logit).to(device=device, dtype=dtype)
        return torch.tensor(
            float(getattr(self.config, "damping_tau", 1.0)),
            device=device,
            dtype=dtype,
        )

    def _feedback_alpha(self, r: int, T_rec: int, ref: torch.Tensor) -> torch.Tensor:
        """Return the damping step size for feedback iteration r."""
        schedule = str(getattr(self.config, "damping_schedule", "constant")).lower()
        dtype = ref.dtype
        device = ref.device

        if schedule == "constant":
            if bool(getattr(self.config, "learn_damping_alpha", False)):
                alpha = torch.sigmoid(self.damping_alpha_logit).to(
                    device=device,
                    dtype=dtype,
                )
            else:
                alpha = torch.tensor(
                    float(getattr(self.config, "damping_alpha", 1.0)),
                    device=device,
                    dtype=dtype,
                )
        else:
            feedback_steps = max(int(T_rec) - 1, 1)
            tau = self._damping_tau(ref)
            if schedule == "normalized":
                alpha = tau / float(feedback_steps)
            elif schedule == "decay":
                beta = float(getattr(self.config, "damping_decay_beta", 0.0))
                idx = torch.arange(feedback_steps, device=device, dtype=torch.float32)
                weights = torch.softmax(-beta * idx, dim=0).to(dtype=dtype)
                alpha = tau * weights[int(r) - 1]
            else:
                raise ValueError(f"Unknown damping_schedule: {schedule}")

        return alpha.clamp(0.0, 1.0)

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
        return_loop_hiddens: bool = False,
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
        token_len = T_
        token_attention_mask = attention_mask

        if self.config.input_emb_norm:
            x = x * (self.config.d_model ** 0.5)
        x = self.transformer.emb_drop(x)

        # Workspace is feedback-only. Keep the token path untouched here so
        # T_rec=1 remains exactly the vanilla split computation.
        m = int(getattr(self.config, "workspace_size", 0))

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

        feedback_eff_bias: Optional[torch.Tensor] = None
        use_workspace = m > 0 and T_rec > 1
        if use_workspace:
            feedback_len = token_len + m
            assert feedback_len <= self.config.max_sequence_length, (
                f"seq_len + workspace_size ({token_len} + {m}) exceeds "
                f"max_sequence_length ({self.config.max_sequence_length}); "
                f"RoPE cache will not cover feedback workspace positions."
            )
            if attention_bias is not None:
                base_bias = attention_bias
                if base_bias.dtype in (torch.int8, torch.bool):
                    base_bias = base_bias.to(dtype=torch.float)
                    base_bias.masked_fill_(
                        base_bias == 0.0, torch.finfo(base_bias.dtype).min
                    )
                base_bias = base_bias[:, :, :token_len, :token_len].to(
                    dtype=torch.float
                )
                feedback_eff_bias = base_bias.new_zeros(
                    base_bias.shape[:-2] + (feedback_len, feedback_len)
                )
                feedback_eff_bias[..., :token_len, :token_len] = base_bias
            elif token_attention_mask is not None and 0.0 in token_attention_mask:
                feedback_eff_bias = self.get_bidirectional_attention_bias(
                    feedback_len, x.device
                ).to(dtype=torch.float)

            if token_attention_mask is not None and 0.0 in token_attention_mask:
                workspace_mask = torch.ones(
                    B_,
                    m,
                    device=token_attention_mask.device,
                    dtype=token_attention_mask.dtype,
                )
                feedback_mask = torch.cat(
                    [token_attention_mask, workspace_mask], dim=1
                )
                feedback_am = feedback_mask.to(dtype=torch.float).view(
                    B_, -1
                )[:, None, None, :]
                feedback_am = (1.0 - feedback_am) * torch.finfo(
                    feedback_am.dtype
                ).min
                if feedback_eff_bias is None:
                    feedback_eff_bias = self.get_bidirectional_attention_bias(
                        feedback_len, x.device
                    ).to(dtype=torch.float)
                feedback_eff_bias = feedback_eff_bias + feedback_am
                ensure_finite_(
                    feedback_eff_bias,
                    check_neg_inf=True,
                    check_pos_inf=False,
                )

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
        diag_raw_residual: List[torch.Tensor] = []
        diag_adapter_update: List[torch.Tensor] = []  # only logged for r >= 1
        diag_damping_alpha: List[torch.Tensor] = []  # only logged for r >= 1
        diag_workspace_norm: List[torch.Tensor] = []  # only logged for r >= 1
        diag_workspace_update: List[torch.Tensor] = []  # only logged for r >= 1
        loop_hiddens: List[torch.Tensor] = []  # for deep supervision (no detach)

        h = e
        prev_h: Optional[torch.Tensor] = None
        use_damped_update = bool(getattr(self.config, "use_damped_update", False))
        learn_damping_alpha = bool(getattr(self.config, "learn_damping_alpha", False))
        learn_damping_tau = bool(getattr(self.config, "learn_damping_tau", True))
        damping_schedule = str(getattr(self.config, "damping_schedule", "constant"))
        workspace: Optional[torch.Tensor] = None
        if use_workspace:
            workspace_slots = self.workspace_slots.unsqueeze(0).expand(
                B_, -1, -1
            ).to(device=h.device, dtype=h.dtype)
            if token_attention_mask is not None:
                weights = token_attention_mask.to(
                    device=h.device, dtype=h.dtype
                ).view(B_, token_len, 1)
                summary = (h * weights).sum(dim=1) / weights.sum(
                    dim=1
                ).clamp_min(1.0)
            else:
                summary = h.mean(dim=1)
            workspace = workspace_slots + summary.unsqueeze(1)

        for r in range(T_rec):
            if r == 0:
                # First pass: no link, no detach.
                h_input = h
            else:
                # Feedback iteration: full BPTT through link + R.
                if self.config.use_latent_feedback:
                    h_input = self.recursive_link(h)
                    if output_loop_diagnostics:
                        with torch.no_grad():
                            update = (h_input - h).norm() / h.norm().clamp_min(1e-6)
                            diag_adapter_update.append(update)
                else:
                    # V1: pure h-recurrence, no link.
                    h_input = h

            # Run R blocks (16 layers, weight-shared across iters).
            with_workspace = r > 0 and workspace is not None
            h_out = (
                torch.cat([h_input, workspace], dim=1)
                if with_workspace
                else h_input
            )
            recurrent_bias = feedback_eff_bias if with_workspace else eff_bias
            for block in blocks[prelude_end:coda_start]:
                h_out, _ = block(
                    h_out,
                    attention_bias=recurrent_bias,
                    layer_past=None,
                    use_cache=False,
                )

            workspace_out: Optional[torch.Tensor] = None
            if with_workspace:
                h_out, workspace_out = h_out[:, :token_len, :], h_out[:, token_len:, :]

            h_next = h_out
            workspace_next = workspace
            if r > 0 and use_damped_update:
                damping_alpha = self._feedback_alpha(r=r, T_rec=T_rec, ref=h)
                h_next = h + damping_alpha * (h_out - h)
                if workspace is not None and workspace_out is not None:
                    workspace_next = workspace + damping_alpha * (
                        workspace_out - workspace
                    )
                if output_loop_diagnostics:
                    diag_damping_alpha.append(damping_alpha.detach())
            elif workspace_out is not None:
                workspace_next = workspace_out

            # Diagnostics (no_grad). Records ||h_r - h_{r-1}|| / ||h_{r-1}||
            # for r >= 1 (i.e., len = T_rec - 1).
            if output_loop_diagnostics and prev_h is not None:
                with torch.no_grad():
                    if use_damped_update:
                        raw_rel = (
                            (h_out - prev_h).norm()
                            / prev_h.norm().clamp_min(1e-6)
                        )
                        diag_raw_residual.append(raw_rel)
                    rel = (h_next - prev_h).norm() / prev_h.norm().clamp_min(1e-6)
                    diag_residual.append(rel)
                    if with_workspace and workspace is not None and workspace_next is not None:
                        diag_workspace_norm.append(
                            workspace_next.norm() / h_next.norm().clamp_min(1e-6)
                        )
                        diag_workspace_update.append(
                            (workspace_next - workspace).norm()
                            / workspace.norm().clamp_min(1e-6)
                        )
            prev_h = h_next

            h = h_next
            workspace = workspace_next

            # Stash post-iteration hidden for deep supervision. No detach
            # (BPTT already retains these for backward); appending a Python
            # reference is free.
            if return_loop_hiddens:
                loop_hiddens.append(h)

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
                    "raw_residual_norm": diag_raw_residual,
                    "adapter_update_norm": diag_adapter_update,
                    "damping_alpha": diag_damping_alpha,
                    "workspace_norm": diag_workspace_norm,
                    "workspace_update_norm": diag_workspace_update,
                    "T_rec": T_rec,
                    "use_damped_update": use_damped_update,
                    "damping_schedule_id": {
                        "constant": 0.0,
                        "normalized": 1.0,
                        "decay": 2.0,
                    }.get(damping_schedule, -1.0),
                    "damping_tau": (
                        self._damping_tau(h).detach()
                        if (
                            use_damped_update
                            and damping_schedule in {"normalized", "decay"}
                        )
                        else float(getattr(self.config, "damping_tau", 1.0))
                    ),
                    "damping_decay_beta": float(
                        getattr(self.config, "damping_decay_beta", 0.0)
                    ),
                    "learn_damping_alpha": learn_damping_alpha,
                    "learn_damping_tau": learn_damping_tau,
                }
                if output_loop_diagnostics
                else None
            ),
            "loop_hiddens": loop_hiddens if return_loop_hiddens else None,
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
                 use_latent_feedback, mu_rec_eval, use_damped_update,
                 damping_schedule, damping_alpha, learn_damping_alpha,
                 damping_tau, learn_damping_tau, damping_decay_beta, workspace_size,
                 workspace_init_std).
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
        return_loop_hiddens: bool = False,
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
            return_loop_hiddens=return_loop_hiddens,
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
            loop_hiddens=loop_extras.get("loop_hiddens"),
        )

    def can_generate(self) -> bool:
        return True
