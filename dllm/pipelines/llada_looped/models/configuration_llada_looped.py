"""
LLaDA-Looped configuration (latent feedback loop).

Extends LLaDAConfig with:
  - prelude/recurrent/coda layer partition
  - latent-feedback toggle (loop variant V2) vs pure h-recurrence (V1)
  - alpha gating scalar (init 0 for drop-in at T_rec=1)
  - recursive-link MLP hidden dim (2-layer residual MLP applied to h_r)

Drop-in property:
    With alpha = 0 and T_rec = 1, the model is numerically equivalent to
    vanilla LLaDA (one sweep through prelude + R + coda blocks). The
    recursive_link's W_2 is also zero-initialized for belt-and-suspenders
    drop-in at any T_rec.

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
        # ---- Loop variant toggle ----
        # V2 (use_latent_feedback=True): apply RecursiveLink (2-layer residual
        #     MLP) to h_r and inject as a gated residual at mask positions.
        # V1 (use_latent_feedback=False): pure h-recurrence h_{r+1} = R(h_r).
        # V0 baseline lives in examples/llada/sft.py (vanilla LLaDA, no loop).
        use_latent_feedback: bool = True,
        # ---- Latent-feedback hyperparameters (V2 only) ----
        # alpha: scalar gate, init 0 -> drop-in at T_rec=1
        alpha_init: float = 0.0,
        # Hidden dim of the 2-layer MLP. None -> defaults to d_model in the
        # model __init__. Common choices: d_model (no expansion, ~2*d^2 params)
        # or 4*d_model (FFN-like expansion, ~8*d^2 params). For LLaDA-8B
        # (d=4096) the no-expansion variant is ~32M params per RecursiveLink.
        recursive_link_hidden_dim: int | None = None,
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

        self.use_latent_feedback = bool(use_latent_feedback)
        self.alpha_init = float(alpha_init)
        self.recursive_link_hidden_dim = (
            int(recursive_link_hidden_dim)
            if recursive_link_hidden_dim is not None
            else None
        )

        self.mu_rec_eval = int(mu_rec_eval)
