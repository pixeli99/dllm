"""
LLaDA-WoDD configuration.

WoDD = Write-on-Demand Diffusion. Extends LLaDA with an append-only latent
memory tape that the model can choose to write to at each inner "thinking
step" of a single denoising forward.

Conceptual frame
----------------
Existing latent-CoT mechanisms for diffusion LMs (MetaState, LATTS, DPad,
Latent Tokens, LaDiR) all use STATE EVOLUTION -- a fixed-capacity working
state that is updated in place across iterations. Empirically they cap at
~5% lift on math reasoning while token CoT scales 30-50%. Hypothesis: the
gap is information-theoretic. Token CoT writes log|V| new bits each step
(append-only); state evolution recycles a fixed-dim buffer. WoDD restores
append-only semantics in latent space.

Mechanism (one denoising forward pass)
--------------------------------------
    1. Embed visible tokens -> h
    2. Run prelude blocks (frozen LLaDA layers)
    3. Initialize tape M = []
    4. For inner step k in 1..T_step:
         a. If M non-empty: h <- h + TapeRead(query=h, kv=M)
         b. h <- recurrent_blocks(h)              # weight-shared across k
         c. pool = masked_mean(h)
         d. g_k = sigmoid(write_gate(pool))       # in [0, 1], soft
         e. v_k = write_content(pool)             # in R^tape_dim
         f. M.append(g_k * v_k)                   # APPEND ONLY
    5. Run coda blocks on final h
    6. Standard LLaDA logits

Soft gating means tape entries with g_k ~ 0 contribute ~0 to cross-attn
output; the tape grows monotonically (append-only) but unused entries
auto-zero. No ST-Gumbel needed -- gradients flow naturally through g_k.

What is NOT WoDD (defends novelty vs nearest neighbors)
-------------------------------------------------------
- MetaState (2603.01331): GRU updates a fixed-size memory IN PLACE.
- LATTS (ICLR 2026): more latent steps, no memory at all.
- DPad (2508.14148): pre-allocated suffix scratch, no per-step writes.
- LaDiR (2510.04573): VAE-encoded thoughts denoised by separate diffusion
  block on FROZEN LLM.
- Coconut: AR feeds hidden state back; not DLM, not append-only.

The append-only semantics + per-step write gate are the WoDD-specific
combination. None of the above is structurally equivalent.
"""

from __future__ import annotations

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


class LLaDAWoDDConfig(LLaDAConfig):
    model_type = "llada_wodd"

    @property
    def effective_n_kv_heads(self) -> int:
        return ModelConfig.effective_n_kv_heads.fget(self)

    def __init__(
        self,
        # ---- Layer partition (same as llada_looped) ----
        prelude_layers: int = 8,
        recurrent_layers: int = 16,
        coda_layers: int = 8,
        # ---- WoDD tape ----
        # Inner thinking steps per denoising forward. T_step=1 makes the tape
        # contain at most one entry; WoDD's append-only character only matters
        # for T_step >= 2.
        t_step_eval: int = 4,
        # Tape entry dimensionality. We project from d_model down so the cross-
        # attention is cheap. Default ~ d_model / 4 to match WriteHead bottleneck.
        tape_dim: int = 1024,
        # Max number of writes per forward. Hard cap so cross-attn KV size is
        # bounded. With multi-slot writes the per-forward tape size is
        # t_step * write_slots, so this must be >= t_step_eval * write_slots.
        tape_max_writes: int = 64,
        # Tape cross-attention heads. Independent from main attention heads so
        # tape attention stays small.
        tape_n_heads: int = 8,
        # ---- Multi-slot write pool ----
        # Number of latent slots written per inner step. S=1 recovers the
        # original single-summary behavior. S>1 lets each step write multiple
        # tape entries chosen by learned queries over the sequence.
        write_slots: int = 16,
        # Heads for the learned-query cross-attention pool.
        write_pool_n_heads: int = 8,
        # ---- Identity-at-init guarantee ----
        # When True, write_content output projection AND tape_read output
        # projection are zero-initialized so the network at step 0 is exactly
        # vanilla split LLaDA. All WoDD gain comes from training.
        zero_init_wodd: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["LLaDAWoDDModelLM"])
        super().__init__(**kwargs)

        self.prelude_layers = int(prelude_layers)
        self.recurrent_layers = int(recurrent_layers)
        self.coda_layers = int(coda_layers)

        self.t_step_eval = int(t_step_eval)
        self.tape_dim = int(tape_dim)
        self.tape_max_writes = int(tape_max_writes)
        self.tape_n_heads = int(tape_n_heads)
        self.write_slots = int(write_slots)
        self.write_pool_n_heads = int(write_pool_n_heads)
        self.zero_init_wodd = bool(zero_init_wodd)

        assert self.t_step_eval >= 1
        assert self.write_slots >= 1
        assert self.tape_max_writes >= self.t_step_eval * self.write_slots, (
            f"tape_max_writes ({self.tape_max_writes}) must be >= "
            f"t_step_eval * write_slots "
            f"({self.t_step_eval} * {self.write_slots} = "
            f"{self.t_step_eval * self.write_slots})"
        )
        assert self.tape_dim % self.tape_n_heads == 0, (
            f"tape_dim ({self.tape_dim}) must be divisible by tape_n_heads "
            f"({self.tape_n_heads})"
        )
        assert self.d_model % self.write_pool_n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"write_pool_n_heads ({self.write_pool_n_heads})"
        )
