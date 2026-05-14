"""Dream-WoDD config (Phase 5 cross-family).

Mirrors LLaDAWoDDConfig structure but extends DreamConfig. The partition
defaults (8/16/8) assume Dream-7B has 32 layers; tighten if Dream's
actual num_hidden_layers differs (Dream-Org/Dream-v0-Instruct-7B has
28 hidden layers — re-tune partition for that variant).
"""
from __future__ import annotations

from dllm.pipelines.dream.models.configuration_dream import DreamConfig


class DreamWoDDConfig(DreamConfig):
    model_type = "dream_wodd"

    def __init__(
        self,
        # Layer partition (sum must be ≤ num_hidden_layers).
        prelude_layers: int = 8,
        recurrent_layers: int = 16,
        coda_layers: int = 8,
        # Tape config (matches LLaDAWoDDConfig defaults).
        t_step_eval: int = 4,
        tape_dim: int = 1024,
        tape_max_writes: int = 16,
        tape_n_heads: int = 8,
        zero_init_wodd: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["DreamWoDDModel"])
        super().__init__(**kwargs)
        self.prelude_layers   = int(prelude_layers)
        self.recurrent_layers = int(recurrent_layers)
        self.coda_layers      = int(coda_layers)
        self.t_step_eval      = int(t_step_eval)
        self.tape_dim         = int(tape_dim)
        self.tape_max_writes  = int(tape_max_writes)
        self.tape_n_heads     = int(tape_n_heads)
        self.zero_init_wodd   = bool(zero_init_wodd)
        assert self.t_step_eval >= 1
        assert self.tape_max_writes >= self.t_step_eval
        assert self.tape_dim % self.tape_n_heads == 0
