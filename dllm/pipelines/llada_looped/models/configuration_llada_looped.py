"""
LLaDA-Looped configuration.

Extends LLaDAConfig with a prelude/recurrent/coda layer split and
Parcae-style middle-loop controls.

Retrofit invariant (μ_rec = 1 must equal vanilla LLaDA):

    h_{r+1} = R( A_disc ⊙ h_r + B_disc · e ),   h_0 = 0,  e = x_prelude

At init we want `h_1 = R(x_prelude)` exactly, i.e. `A_disc ⊙ 0 + B_disc · e == e`.
That requires `B_disc = I` at init (we ignore `A` entirely while `h_0 = 0`).
So:

    A_init_log = 5.0   -> A_disc = exp(-exp(5)·1) ≈ 0      (irrelevant initially,
                                                            keeps state decayed)
    delta_init = 1.0
    B_init_identity = True  -> B_inject = I (full) or 1-vector (diag).
    use_input_norm  = False -> e = x_prelude verbatim (no LN-induced rescale).
"""

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


class LLaDALoopedConfig(LLaDAConfig):
    model_type = "llada_looped"

    # ModelConfig defines this as a @property; LLaDAConfig inherits from
    # PretrainedConfig (not ModelConfig), so the property doesn't come along.
    # Vanilla LLaDA sidesteps this via create_model_config_from_pretrained_config;
    # we pass LLaDALoopedConfig straight through to LLaDAModel, so we need it here.
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
        # ---- SSM stabilizer (Parcae-style) ----
        # e = x_prelude (verbatim) when False; LayerNorm(x_prelude) when True.
        # Default False so μ_rec=1 is exactly vanilla LLaDA.
        use_input_norm: bool = False,
        # Diagonal vs full B_inject matrix.
        use_diag_B: bool = False,
        # log_A = 5 -> A_cont = -exp(5) = -148, A_disc = exp(-148) ≈ 0.
        # With h_0 = 0, A_disc ⊙ h_0 = 0 regardless, so init value of A
        # only matters once h starts taking non-zero values across iterations.
        A_init_log: float = 5.0,
        delta_init: float = 1.0,
        # Identity init for B so B·e == e at step 0: μ_rec=1 is drop-in.
        B_init_identity: bool = True,
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
        self.B_init_identity = B_init_identity
