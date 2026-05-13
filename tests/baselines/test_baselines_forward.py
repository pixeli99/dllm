"""Smoke + invariant tests for the Phase-2 baseline pipelines.

We verify, at toy scale (d=64, 4 layers):

  - **LATTS** (A1): zero trainable adapter; T_step=1 logits MATCH vanilla
    LLaDA exactly. T_step=2 logits DIFFER (recurrent block applied twice).
  - **MetaState** (A2): zero_init_metastate=True makes T_step=1 logits
    MATCH vanilla LLaDA exactly (out_proj=0 and GRU z-bias=-5 -> state
    contribution to h is 0). T_step=2 differs.
  - **DPad** (A3): the strict drop-in invariant does NOT hold (attaching
    K scratch positions changes softmax normalisation). We only check
    that the forward is finite and that changing the scratchpad changes
    the logits at original positions.
  - All three accept `from_llada_checkpoint`-style retrofit kwargs in
    their config and can do a single SGD step without errors.

These tests run with `pytest tests/baselines/` and currently SKIP when
dllm cannot import (the same guard pattern as tests/wodd/).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

try:
    import dllm  # noqa: F401
    from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.llada_latts import (
        LLaDALATTSConfig, LLaDALATTSModelLM,
    )
    from dllm.pipelines.llada_metastate import (
        LLaDAMetaStateConfig, LLaDAMetaStateModelLM,
    )
    from dllm.pipelines.llada_dpad import (
        LLaDADPadConfig, LLaDADPadModelLM,
    )
    HAS_DLLM = True
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    HAS_DLLM = False
    _IMPORT_ERROR = e


pytestmark = pytest.mark.skipif(
    not HAS_DLLM,
    reason=f"baseline pipelines not importable: {_IMPORT_ERROR}",
)


# ---------------------------------------------------------------------------
# Shared toy LLaDA config (matches tests/wodd toy size)
# ---------------------------------------------------------------------------

_TOY_BACKBONE = dict(
    d_model=64, n_heads=2, n_layers=4, n_kv_heads=2,
    mlp_hidden_size=128, max_sequence_length=64,
    vocab_size=128, embedding_size=128,
    mask_token_id=126, pad_token_id=127, eos_token_id=125,
    rope=True, alibi=False, block_group_size=1, block_type="llama",
    activation_type="silu",
    weight_tying=True, scale_logits=False, input_emb_norm=False,
    layer_norm_type="default",
    attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
    init_device="cpu",
)


def _build_pair(pipeline_cls, config_cls, extra_kwargs, seed=0):
    """Build a (vanilla, pipeline_model) pair with shared weights.
    Both models are torch.float32; `pipeline_model` has had vanilla's
    state_dict copied in via strict=False.
    """
    torch.manual_seed(seed)
    v_cfg = LLaDAConfig(**_TOY_BACKBONE)
    v = LLaDAModelLM(v_cfg, init_params=True).to(dtype=torch.float32).eval()
    torch.manual_seed(seed)
    p_cfg = config_cls(**_TOY_BACKBONE, **extra_kwargs)
    p = pipeline_cls(p_cfg, init_params=True).to(dtype=torch.float32).eval()
    p.load_state_dict(v.state_dict(), strict=False)
    return v, p


def _dev(m):
    return next(m.parameters()).device


# ---------------------------------------------------------------------------
# A1: LATTS — zero trainable adapter
# ---------------------------------------------------------------------------

def test_latts_t1_matches_vanilla_exactly():
    v, p = _build_pair(
        LLaDALATTSModelLM, LLaDALATTSConfig,
        dict(prelude_layers=1, recurrent_layers=2, coda_layers=1, t_step_eval=2),
    )
    dev = _dev(p)
    ids = torch.randint(0, 100, (2, 8), device=dev)
    am  = torch.ones(2, 8, dtype=torch.long, device=dev)
    with torch.no_grad():
        a = v(input_ids=ids, attention_mask=am).logits
        b = p(input_ids=ids, attention_mask=am, T_step=1).logits
    assert (a - b).abs().max().item() == 0.0


def test_latts_t2_differs_and_finite():
    v, p = _build_pair(
        LLaDALATTSModelLM, LLaDALATTSConfig,
        dict(prelude_layers=1, recurrent_layers=2, coda_layers=1, t_step_eval=2),
    )
    dev = _dev(p)
    ids = torch.randint(0, 100, (2, 8), device=dev)
    am  = torch.ones(2, 8, dtype=torch.long, device=dev)
    with torch.no_grad():
        a = v(input_ids=ids, attention_mask=am).logits
        b = p(input_ids=ids, attention_mask=am, T_step=2).logits
    assert torch.isfinite(b).all()
    assert (a - b).abs().max().item() > 1e-4


# ---------------------------------------------------------------------------
# A2: MetaState — state-evolution baseline
# ---------------------------------------------------------------------------

def test_metastate_t1_matches_vanilla_exactly():
    """Zero-init: out_proj=0 and GRU z-bias=-5 -> state contribution to
    h is 0, so T_step=1 is byte-identical to vanilla LLaDA."""
    v, p = _build_pair(
        LLaDAMetaStateModelLM, LLaDAMetaStateConfig,
        dict(
            prelude_layers=1, recurrent_layers=2, coda_layers=1, t_step_eval=2,
            state_slots=8, state_dim=16, state_n_heads=2,
            zero_init_metastate=True,
        ),
    )
    dev = _dev(p)
    ids = torch.randint(0, 100, (2, 8), device=dev)
    am  = torch.ones(2, 8, dtype=torch.long, device=dev)
    with torch.no_grad():
        a = v(input_ids=ids, attention_mask=am).logits
        b = p(input_ids=ids, attention_mask=am, T_step=1).logits
    assert (a - b).abs().max().item() == 0.0


def test_metastate_t2_differs_and_finite():
    v, p = _build_pair(
        LLaDAMetaStateModelLM, LLaDAMetaStateConfig,
        dict(
            prelude_layers=1, recurrent_layers=2, coda_layers=1, t_step_eval=2,
            state_slots=8, state_dim=16, state_n_heads=2,
            zero_init_metastate=True,
        ),
    )
    dev = _dev(p)
    ids = torch.randint(0, 100, (2, 8), device=dev)
    am  = torch.ones(2, 8, dtype=torch.long, device=dev)
    with torch.no_grad():
        a = v(input_ids=ids, attention_mask=am).logits
        b = p(input_ids=ids, attention_mask=am, T_step=2).logits
    assert torch.isfinite(b).all()
    assert (a - b).abs().max().item() > 1e-4


def test_metastate_state_param_count_independent_of_slots():
    """Phase 3 capacity-sweep design: increasing state_slots adds only
    `state_init.state` params (M*D), not the GRU/StateRead params, so
    the trainable parameter count varies cleanly with M."""
    counts = {}
    for M in (8, 16, 32):
        torch.manual_seed(0)
        cfg = LLaDAMetaStateConfig(
            prelude_layers=1, recurrent_layers=2, coda_layers=1, t_step_eval=2,
            state_slots=M, state_dim=16, state_n_heads=2,
            zero_init_metastate=True, **_TOY_BACKBONE,
        )
        m = LLaDAMetaStateModelLM(cfg, init_params=True).eval()
        counts[M] = {
            "state_init":  m.model.state_init.state.numel(),
            "state_read":  sum(p.numel() for p in m.model.state_read.parameters()),
            "state_update": sum(p.numel() for p in m.model.state_update.parameters()),
        }
    # state_init scales linearly with M, state_read / state_update do not.
    assert counts[8]["state_read"]  == counts[16]["state_read"]  == counts[32]["state_read"]
    assert counts[8]["state_update"] == counts[16]["state_update"] == counts[32]["state_update"]
    assert counts[8]["state_init"]  < counts[16]["state_init"]  < counts[32]["state_init"]


# ---------------------------------------------------------------------------
# A3: DPad — suffix scratchpad
# ---------------------------------------------------------------------------

def test_dpad_zero_scratch_drift_bounded():
    """DPad does NOT have a strict drop-in invariant: attaching K extra
    positions changes the softmax denominator at every original position
    even when scratchpad K/V are zero. We assert only that the drift is
    finite and modest."""
    v, p = _build_pair(
        LLaDADPadModelLM, LLaDADPadConfig,
        dict(prelude_layers=1, recurrent_layers=2, coda_layers=1, dpad_len=8,
             zero_init_dpad=True),
    )
    dev = _dev(p)
    ids = torch.randint(0, 100, (2, 8), device=dev)
    am  = torch.ones(2, 8, dtype=torch.long, device=dev)
    with torch.no_grad():
        a = v(input_ids=ids, attention_mask=am).logits
        b = p(input_ids=ids, attention_mask=am).logits
    assert torch.isfinite(b).all()
    # The drift should be bounded -- not arbitrarily large; just from
    # softmax renormalisation over K extra zero-key positions.
    drift = (a - b).abs().max().item()
    assert 0.0 < drift < 5.0, f"unexpected DPad init drift: {drift}"


def test_dpad_random_scratch_changes_logits():
    v, p = _build_pair(
        LLaDADPadModelLM, LLaDADPadConfig,
        dict(prelude_layers=1, recurrent_layers=2, coda_layers=1, dpad_len=8,
             zero_init_dpad=True),
    )
    dev = _dev(p)
    ids = torch.randint(0, 100, (2, 8), device=dev)
    am  = torch.ones(2, 8, dtype=torch.long, device=dev)
    with torch.no_grad():
        b0 = p(input_ids=ids, attention_mask=am).logits
        p.model.scratchpad.copy_(torch.randn_like(p.model.scratchpad) * 0.1)
        b1 = p(input_ids=ids, attention_mask=am).logits
    assert (b0 - b1).abs().max().item() > 1e-4


# ---------------------------------------------------------------------------
# Backward compatibility: one SGD step on each pipeline (sanity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cls_cfg,cls_lm,extra,kw", [
    (
        "latts", LLaDALATTSConfig, LLaDALATTSModelLM,
        dict(prelude_layers=1, recurrent_layers=2, coda_layers=1, t_step_eval=2),
        dict(T_step=2),
    ),
    (
        "metastate", LLaDAMetaStateConfig, LLaDAMetaStateModelLM,
        dict(prelude_layers=1, recurrent_layers=2, coda_layers=1, t_step_eval=2,
             state_slots=4, state_dim=16, state_n_heads=2,
             zero_init_metastate=True),
        dict(T_step=2),
    ),
    (
        "dpad", LLaDADPadConfig, LLaDADPadModelLM,
        dict(prelude_layers=1, recurrent_layers=2, coda_layers=1, dpad_len=4,
             zero_init_dpad=True),
        dict(),
    ),
])
def test_one_sgd_step(name, cls_cfg, cls_lm, extra, kw):
    """One forward+backward+optim step should yield finite gradients on
    the pipeline-specific trainable params."""
    torch.manual_seed(0)
    cfg = cls_cfg(**_TOY_BACKBONE, **extra)
    m = cls_lm(cfg, init_params=True).to(dtype=torch.float32).train()
    dev = _dev(m)
    B, T = 2, 6
    ids   = torch.randint(0, 100, (B, T), device=dev)
    am    = torch.ones(B, T, dtype=torch.long, device=dev)
    label = ids.clone()
    label[:, :2] = -100

    out = m(input_ids=ids, attention_mask=am, **kw)
    logits = out.logits
    valid  = label != -100
    loss   = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), label.clamp(min=0), reduction="none"
    )
    loss   = (loss * valid.float()).sum() / valid.sum().clamp_min(1)
    assert torch.isfinite(loss).all(), f"{name}: non-finite loss"
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    opt.zero_grad()
    loss.backward()
    # Every trainable param should have a finite grad.
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name}: non-finite grad in {n}"
    opt.step()
