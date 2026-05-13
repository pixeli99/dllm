"""Unit tests for WoDD building blocks.

These tests bypass `dllm/__init__.py` so they run in stripped-down local
environments where heavy optional deps are missing -- same pattern as
tests/td/test_reentry_modules.py.

We test six properties of the WoDD primitives that, taken together,
constitute the formal claims of the design:

  1. WriteHead at zero-init produces v_k = 0 (so step-0 tape is no-op).
  2. WriteHead gate is in [0, 1] and starts near gate_bias_init.
  3. TapeRead at zero-init outputs all-zero (so step-0 cross-attn is no-op).
  4. TapeRead handles K > 1 entries without shape errors and respects
     position embeddings (different write positions produce different K/V).
  5. MemoryTape is append-only -- no .overwrite, max_writes enforced.
  6. End-to-end gradient flow from a downstream loss through tape gate.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_wodd_modules():
    name = "_wodd_modules_test_view"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(
        _REPO_ROOT,
        "dllm",
        "pipelines",
        "llada_wodd",
        "models",
        "_wodd_modules.py",
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mods():
    return _load_wodd_modules()


# ---------------------------------------------------------------------------
# masked_mean
# ---------------------------------------------------------------------------

def test_masked_mean_no_mask(mods):
    h = torch.randn(2, 5, 8)
    out = mods.masked_mean(h, attention_mask=None)
    assert out.shape == (2, 8)
    assert torch.allclose(out, h.mean(dim=1), atol=1e-6)


def test_masked_mean_with_pad(mods):
    h = torch.ones(1, 4, 3)
    h[0, 2:] = 99.0  # padded positions
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    out = mods.masked_mean(h, attention_mask=attention_mask)
    # Only positions 0,1 (all 1s) contribute.
    assert torch.allclose(out, torch.ones(1, 3))


# ---------------------------------------------------------------------------
# WriteHead
# ---------------------------------------------------------------------------

def test_write_head_zero_init_content_zero(mods):
    """zero_init=True => content output is exactly 0."""
    torch.manual_seed(0)
    head = mods.WriteHead(d_model=16, tape_dim=8, zero_init=True)
    pool = torch.randn(3, 16)
    g, v = head(pool)
    assert v.shape == (3, 8)
    # Zero-init: content_proj.weight=0, content_proj.bias=0 -> output = 0.
    assert torch.equal(v, torch.zeros_like(v))


def test_write_head_gate_in_unit_interval(mods):
    torch.manual_seed(0)
    head = mods.WriteHead(d_model=16, tape_dim=8, gate_bias_init=-2.0)
    pool = torch.randn(4, 16)
    g, v = head(pool)
    assert g.shape == (4, 1)
    assert torch.all(g >= 0) and torch.all(g <= 1)


def test_write_head_gate_starts_near_sigmoid_bias(mods):
    """With zero_init weights -> only the bias drives the gate logit."""
    head = mods.WriteHead(d_model=8, tape_dim=4, zero_init=True, gate_bias_init=-2.0)
    # Zero out gate_proj.weight too so the gate ONLY depends on bias.
    with torch.no_grad():
        head.gate_proj.weight.zero_()
    pool = torch.randn(5, 8)
    g, _ = head(pool)
    expected = torch.sigmoid(torch.tensor(-2.0))
    assert torch.allclose(g.squeeze(-1), expected.expand(5), atol=1e-5)


# ---------------------------------------------------------------------------
# MemoryTape
# ---------------------------------------------------------------------------

def test_memory_tape_append_only(mods):
    tape = mods.MemoryTape(batch_size=2, tape_dim=4, max_writes=3)
    assert tape.is_empty()
    v1 = torch.randn(2, 4)
    g1 = torch.rand(2, 1)
    tape.append(v1, gate=g1)
    assert len(tape) == 1
    assert not tape.is_empty()
    assert torch.equal(tape.stack().squeeze(1), v1)

    # No .overwrite method exists.
    assert not hasattr(tape, "overwrite")
    assert not hasattr(tape, "__setitem__")


def test_memory_tape_max_writes_enforced(mods):
    tape = mods.MemoryTape(batch_size=1, tape_dim=2, max_writes=2)
    for _ in range(2):
        tape.append(torch.randn(1, 2), gate=torch.rand(1, 1))
    with pytest.raises(RuntimeError, match="full"):
        tape.append(torch.randn(1, 2), gate=torch.rand(1, 1))


def test_memory_tape_stack_after_k_appends(mods):
    tape = mods.MemoryTape(batch_size=3, tape_dim=5, max_writes=4)
    entries = [torch.randn(3, 5) for _ in range(4)]
    gates = [torch.rand(3, 1) for _ in range(4)]
    for v, g in zip(entries, gates):
        tape.append(v, gate=g)
    stacked = tape.stack()
    assert stacked.shape == (3, 4, 5)
    for k, v in enumerate(entries):
        assert torch.equal(stacked[:, k], v)
    gate_stack = tape.gates()
    assert gate_stack.shape == (3, 4, 1)


# ---------------------------------------------------------------------------
# TapeRead
# ---------------------------------------------------------------------------

def test_tape_read_zero_init_outputs_zero(mods):
    """out_proj zero-init => output is exactly 0 regardless of inputs."""
    torch.manual_seed(0)
    read = mods.TapeRead(
        d_model=16, tape_dim=8, max_writes=4, n_heads=2, zero_init=True
    )
    h = torch.randn(2, 5, 16)
    tape = torch.randn(2, 3, 8)
    out = read(h, tape)
    assert out.shape == (2, 5, 16)
    assert torch.equal(out, torch.zeros_like(out))


def test_tape_read_shape_and_gradient(mods):
    torch.manual_seed(0)
    read = mods.TapeRead(
        d_model=16, tape_dim=8, max_writes=4, n_heads=2, zero_init=False
    )
    h = torch.randn(2, 5, 16, requires_grad=True)
    tape = torch.randn(2, 3, 8, requires_grad=True)
    out = read(h, tape)
    assert out.shape == (2, 5, 16)

    loss = out.sum()
    loss.backward()
    assert h.grad is not None and tape.grad is not None
    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(tape.grad).all()


def test_tape_read_respects_write_position(mods):
    """Same tape entry at different write positions => different cross-attn."""
    torch.manual_seed(0)
    read = mods.TapeRead(
        d_model=8, tape_dim=8, max_writes=4, n_heads=2, zero_init=False
    )
    h = torch.randn(1, 3, 8)
    same_entry = torch.randn(1, 1, 8)

    # Tape of length 1 vs length 2 (with new entry at slot index 1).
    tape_len1 = same_entry
    tape_len2 = torch.cat([same_entry, same_entry], dim=1)

    out1 = read(h, tape_len1)
    out2 = read(h, tape_len2)
    # Because position embedding differs between slot 0 and slot 1, the
    # second tape entry contributes differently than the first.
    assert not torch.allclose(out1, out2, atol=1e-3)


def test_tape_read_attention_mask_zeros_pad_positions(mods):
    """Output at padded positions should be exactly 0 when mask passed."""
    torch.manual_seed(0)
    read = mods.TapeRead(
        d_model=8, tape_dim=8, max_writes=4, n_heads=2, zero_init=False
    )
    h = torch.randn(1, 4, 8)
    tape = torch.randn(1, 2, 8)
    attn = torch.tensor([[1, 1, 0, 0]])
    out = read(h, tape, attention_mask=attn)
    # Padded positions 2,3 are zeroed.
    assert torch.equal(out[0, 2:], torch.zeros(2, 8))


# ---------------------------------------------------------------------------
# End-to-end gradient flow through WriteHead -> tape -> downstream loss
# ---------------------------------------------------------------------------

def test_gradient_flows_through_gate(mods):
    """Loss differentiable w.r.t. WriteHead.gate_proj when gate gates content.

    Setup: pool -> WriteHead -> (g, v) -> tape entry = g*v -> TapeRead -> sum.
    """
    torch.manual_seed(0)
    head = mods.WriteHead(d_model=16, tape_dim=8, zero_init=False)
    read = mods.TapeRead(
        d_model=16, tape_dim=8, max_writes=4, n_heads=2, zero_init=False
    )
    pool = torch.randn(2, 16)
    h = torch.randn(2, 5, 16)
    g, v = head(pool)
    entry = (g * v).unsqueeze(1)  # (B, 1, tape_dim)
    out = read(h, entry)
    loss = out.sum()
    loss.backward()

    # Gradient should flow to gate_proj.weight (via g) AND content_proj.weight
    # (via v), because entry = g * v depends on both.
    assert head.gate_proj.weight.grad is not None
    assert torch.isfinite(head.gate_proj.weight.grad).all()
    assert head.gate_proj.weight.grad.abs().sum() > 0

    assert head.content_proj.weight.grad is not None
    assert torch.isfinite(head.content_proj.weight.grad).all()
    assert head.content_proj.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# WoDDStepStats helpers
# ---------------------------------------------------------------------------

def test_step_stats_entropy_zero_at_boundary(mods):
    """Bernoulli entropy is ~0 at g=0 and ~0 at g=1."""
    g_low = torch.full((1, 3, 1), 1e-6)
    g_high = torch.full((1, 3, 1), 1.0 - 1e-6)
    s_low = mods.WoDDStepStats(write_gates=g_low)
    s_high = mods.WoDDStepStats(write_gates=g_high)
    assert s_low.gate_entropy.abs().max() < 5e-5
    assert s_high.gate_entropy.abs().max() < 5e-5


def test_step_stats_entropy_max_at_half(mods):
    """Bernoulli entropy max ln(2) ~ 0.693 at g=0.5."""
    g = torch.full((1, 3, 1), 0.5)
    s = mods.WoDDStepStats(write_gates=g)
    assert torch.allclose(s.gate_entropy, torch.full((1,), torch.log(torch.tensor(2.0))), atol=1e-5)


def test_step_stats_mean_gate(mods):
    g = torch.tensor([[[0.0], [0.5], [1.0]], [[0.25], [0.5], [0.75]]])
    s = mods.WoDDStepStats(write_gates=g)
    assert torch.allclose(s.mean_gate, torch.tensor([0.5, 0.5]))
