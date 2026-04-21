"""
LLaDA-Looped configuration.

Extends LLaDAConfig with a prelude/recurrent/coda layer split and
Parcae-style middle-loop controls. At `mu_rec_eval=1` with zero-init
`B_inject` and `A_init_log` that makes `A_disc ≈ I`, the forward is
numerically equivalent (up to a near-identity input-LayerNorm) to
vanilla LLaDA, so retrofit from a pretrained LLaDA-8B checkpoint is
a drop-in weight load.

Run:
    # This file is a config module — no entry point. See:
    #   /Users/pixeli/dllm/examples/llada_looped/sft.py
"""

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


class LLaDALoopedConfig(LLaDAConfig):
    model_type = "llada_looped"

    # ModelConfig defines this as a @property, not a dataclass field. LLaDAConfig
    # inherits from PretrainedConfig (not ModelConfig), so the property doesn't
    # come along. Vanilla LLaDA sidesteps this because LLaDAModelLM.__init__
    # converts LLaDAConfig -> ModelConfig via create_model_config_from_pretrained_config
    # before constructing LLaDAModel. We pass LLaDALoopedConfig straight through
    # to LLaDAModel, so we need the property here directly.
    @property
    def effective_n_kv_heads(self) -> int:
        return ModelConfig.effective_n_kv_heads.fget(self)

    def __init__(
        self,
        # ---- Layer partition ----
        prelude_layers: int = 8,
        recurrent_layers: int = 16,
        coda_layers: int = 8,
        # ---- Test-time recurrence default ----
        mu_rec_eval: int = 2,
        # ---- Parcae stabilizer ----
        use_input_norm: bool = True,
        use_diag_B: bool = False,
        # log_A is initialized so that A_disc = exp(-exp(log_A) * delta) ≈ 1.
        # exp(-4) ≈ 0.018 -> A_disc ≈ exp(-0.018) ≈ 0.982.
        A_init_log: float = -4.0,
        delta_init: float = 1.0,
        # B_inject init: 0 -> loop is identity w.r.t. vanilla LLaDA at μ_rec=1.
        B_init_scale: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["LLaDALoopedModelLM"])
        super().__init__(**kwargs)

        total = prelude_layers + recurrent_layers + coda_layers
        assert total <= self.n_layers, (
            f"prelude ({prelude_layers}) + recurrent ({recurrent_layers}) + "
            f"coda ({coda_layers}) = {total} exceeds n_layers ({self.n_layers})."
        )
        assert recurrent_layers >= 1, "recurrent_layers must be >= 1."

        self.prelude_layers = prelude_layers
        self.recurrent_layers = recurrent_layers
        self.coda_layers = coda_layers
        self.mu_rec_eval = mu_rec_eval
        self.use_input_norm = use_input_norm
        self.use_diag_B = use_diag_B
        self.A_init_log = A_init_log
        self.delta_init = delta_init
        self.B_init_scale = B_init_scale
