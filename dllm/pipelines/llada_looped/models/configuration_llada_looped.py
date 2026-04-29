"""
LLaDA-Looped configuration (belief-bottleneck loop).

Extends LLaDAConfig with:
  - prelude/recurrent/coda layer partition
  - belief-feedback toggle (V2) vs pure h-recurrence (V1)
  - alpha gating scalar (init 0 for drop-in at T_rec=1)
  - tau temperature for soft belief (FIXED, not learned -- it is a
    framing-defined property of the soft-belief mechanism)
  - top-k truncation of `belief @ embed_matrix`
  - RMSNorm magnitude alignment on belief embedding (residual-stream-
    hypothesis-friendly substitute for a learned input-space projection)

Drop-in property:
    With alpha = 0 and T_rec = 1, the model is numerically equivalent to
    vanilla LLaDA (one sweep through prelude + R + coda blocks).
    For T_rec > 1 with alpha = 0, the loop applies R T_rec times without
    belief feedback -- this is V1 (pure h-recurrence), used as ablation.

Run:
    # This module is imported by the loaded model. See:
    #   /Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py
"""

from __future__ import annotations

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


class LLaDALoopedConfig(LLaDAConfig):
    model_type = "llada_looped"

    # `effective_n_kv_heads` is defined as a @property on ModelConfig, but
    # LLaDAConfig inherits from PretrainedConfig directly. Vanilla LLaDA
    # bridges via create_model_config_from_pretrained_config; we pass this
    # config straight through to LLaDAModel, so we need to expose the
    # property here.
    @property
    def effective_n_kv_heads(self) -> int:
        return ModelConfig.effective_n_kv_heads.fget(self)

    def __init__(
        self,
        # ---- Layer partition ----
        prelude_layers: int = 8,
        recurrent_layers: int = 16,
        coda_layers: int = 8,
        # ---- Loop variant toggles ----
        # V2 (use_belief_feedback=True): readout -> soft belief -> embed -> gated injection
        # V1 (use_belief_feedback=False): pure h-recurrence h_{r+1} = R(h_r), no feedback
        # V0 baseline lives in examples/llada/sft.py (vanilla LLaDA, no loop module needed).
        use_belief_feedback: bool = True,
        # ---- Belief-feedback hyperparameters (only used when use_belief_feedback=True) ----
        # alpha: scalar gate, init 0 -> drop-in at T_rec=1.
        alpha_init: float = 0.0,
        # tau: temperature, FIXED (not learnable). Choose so softmax(logits/tau)
        # has effective support of ~10-30 tokens. With LLaDA-8B's typical
        # logit norms, tau ~ 1.0 to ~5.0 is the soft regime; measure logit
        # scale empirically and set accordingly.
        tau: float = 1.0,
        # Top-k truncation for belief @ embed_matrix; saves O(B*L*V*d) memory.
        # Equivalent to full softmax-then-matmul up to numerical noise when
        # softmax mass concentrated on top-k.
        belief_top_k: int = 32,
        # RMSNorm magnitude alignment on belief embedding before gated
        # injection. Residual-stream hypothesis says input-embed and h_23
        # share basis up to magnitude; RMSNorm fixes the magnitude mismatch
        # without a learned d->d projection.
        use_belief_norm: bool = True,
        # ---- Eval-time T_rec default ----
        mu_rec_eval: int = 4,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["LLaDALoopedModelLM"])
        super().__init__(**kwargs)

        # NB: validity of (prelude + recurrent + coda) <= n_layers is checked
        # in LLaDALoopedModel.__init__, not here. Reason: HF's to_diff_dict
        # serialization does `self.__class__()` with no kwargs and would
        # otherwise crash on n_layers=12 default during TensorBoard logging.

        self.prelude_layers = int(prelude_layers)
        self.recurrent_layers = int(recurrent_layers)
        self.coda_layers = int(coda_layers)

        self.use_belief_feedback = bool(use_belief_feedback)
        self.alpha_init = float(alpha_init)
        self.tau = float(tau)
        self.belief_top_k = int(belief_top_k)
        self.use_belief_norm = bool(use_belief_norm)

        self.mu_rec_eval = int(mu_rec_eval)
