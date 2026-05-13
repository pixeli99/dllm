"""LLaDA-DPad configuration.

Suffix-scratchpad baseline. We append K extra learnable positions to the
sequence; the model runs the standard 32-block LLaDA forward over the
extended sequence and outputs logits only at the original positions.

T_step is *not* a config knob here -- the design is single-pass. Layer
partition (prelude/recurrent/coda) is kept in the config for diagnostic
parity with the other arms but is unused at forward time (we run all
blocks once over [seq; scratchpad]).
"""
from __future__ import annotations

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, ModelConfig


class LLaDADPadConfig(LLaDAConfig):
    model_type = "llada_dpad"

    @property
    def effective_n_kv_heads(self) -> int:
        return ModelConfig.effective_n_kv_heads.fget(self)

    def __init__(
        self,
        # Matched layer partition for honest cross-arm bookkeeping.
        prelude_layers: int = 8,
        recurrent_layers: int = 16,
        coda_layers: int = 8,
        # Number of scratchpad positions appended to the sequence.
        # WODD_PLAN.md §3.2 matched-FLOPs target: choose K so attention
        # cost roughly equals WoDD's T_step·rec compute. K=512 with
        # seq_len 512 ~ doubles attention; tune per cluster experiment.
        dpad_len: int = 512,
        # When True, scratchpad embeddings are zero-init'd. NB this is
        # NOT a strict drop-in invariant for DPad: even with K=V=0 at
        # scratch positions, attaching K extra positions to the sequence
        # changes the softmax denominator at every original position,
        # so the original-position logits differ from vanilla LLaDA by
        # O(K/T) relative magnitude. Smoke test at toy scale shows ~0.3
        # max-abs logit drift at init (T=8, K=8). Training is expected
        # to absorb this. The MetaState/WoDD invariant claim does not
        # apply to DPad by design.
        zero_init_dpad: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["LLaDADPadModelLM"])
        super().__init__(**kwargs)
        self.prelude_layers   = int(prelude_layers)
        self.recurrent_layers = int(recurrent_layers)
        self.coda_layers      = int(coda_layers)
        self.dpad_len         = int(dpad_len)
        self.zero_init_dpad   = bool(zero_init_dpad)
        assert self.dpad_len >= 0
