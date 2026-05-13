"""WoDD invariant tests that DON'T depend on the full LLaDA model.

We construct a mock 'recurrent block' (a tiny LayerNorm + MLP) and a mock
'prelude/coda' (identity passthrough), then exercise the WoDD inner-loop
logic end-to-end:

    h0 = embed
    M = []
    for k in 0..T-1:
        if M non-empty: h <- h + TapeRead(h, M)
        h <- mock_recurrent_block(h)
        (g, v) = WriteHead(masked_mean(h))
        M.append(g * v, gate=g)
    out = mock_coda(h)

This validates:

  - End-to-end gradient flow through the inner-loop chain across T_step iters.
  - The "T_step=1" boundary case has no tape read (tape is empty before
    the first recurrent application).
  - Zero-init invariant: tape contributions are 0 at init -> output equals
    'recurrent block applied T_step times on the same input', i.e. the
    WoDD model degenerates to plain weight-sharing recurrence at init.
  - Append-only contract holds across iterations.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_wodd_modules():
    name = "_wodd_modules_invariant_view"
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
# Mock recurrent block
# ---------------------------------------------------------------------------

class MockRecurrentBlock(nn.Module):
    """Stand-in for the 16 LLaDA blocks. Just LN + MLP residual."""
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x):
        return x + self.fc(self.ln(x))


def _wodd_forward(
    mods,
    h0,
    recurrent_block,
    write_head,
    tape_read,
    T_step,
    attention_mask=None,
    max_writes=8,
):
    """Mini WoDD inner-loop, mirroring LLaDAWoDDModel.forward."""
    B, T, D = h0.shape
    tape_dim = write_head.content_proj.out_features

    tape = mods.MemoryTape(batch_size=B, tape_dim=tape_dim, max_writes=max_writes)
    h = h0
    for k in range(T_step):
        if not tape.is_empty():
            tape_stack = tape.stack()
            h = h + tape_read(h, tape_stack, attention_mask=attention_mask)
        h = recurrent_block(h)
        pool = mods.masked_mean(h, attention_mask)
        g, v = write_head(pool)
        tape.append(g * v, gate=g)
    return h, tape


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t_step_1_no_tape_read(mods):
    """T_step=1: tape is empty at the start; the recurrent block runs once;
    a single write happens at the end. No tape read called."""
    torch.manual_seed(0)
    D = 32
    h0 = torch.randn(2, 4, D)
    rec = MockRecurrentBlock(D)
    write = mods.WriteHead(d_model=D, tape_dim=16, zero_init=False)
    tread = mods.TapeRead(
        d_model=D, tape_dim=16, max_writes=4, n_heads=2, zero_init=False
    )

    h_out, tape = _wodd_forward(mods, h0, rec, write, tread, T_step=1)
    # No tape read happened, so h_out should match a plain rec(h0).
    expected = rec(h0)
    assert torch.allclose(h_out, expected, atol=1e-6)
    assert len(tape) == 1


def test_zero_init_degenerates_to_pure_weight_sharing(mods):
    """With zero_init_wodd=True, tape_read output is 0 throughout, so the
    WoDD model behaves identically to "recurrent block applied T_step times"
    (no feedback bridge). This is the formal "WoDD adds nothing at init"
    claim.
    """
    torch.manual_seed(0)
    D = 32
    h0 = torch.randn(2, 4, D)
    rec = MockRecurrentBlock(D)
    write = mods.WriteHead(d_model=D, tape_dim=16, zero_init=True)
    tread = mods.TapeRead(
        d_model=D, tape_dim=16, max_writes=4, n_heads=2, zero_init=True
    )

    T_step = 3
    h_wodd, _ = _wodd_forward(mods, h0, rec, write, tread, T_step=T_step)

    # Plain recurrent (no tape) reference: rec applied T_step times.
    h_ref = h0
    for _ in range(T_step):
        h_ref = rec(h_ref)
    assert torch.allclose(h_wodd, h_ref, atol=1e-6), (
        "Zero-init WoDD should be bit-equivalent to plain T_step-fold "
        "recurrent application."
    )


def test_grad_flow_across_all_iterations(mods):
    """Gradient from final loss should reach the FIRST iteration's gate
    (proves end-to-end BPTT through the inner loop).
    """
    torch.manual_seed(0)
    D = 16
    h0 = torch.randn(1, 3, D, requires_grad=True)
    rec = MockRecurrentBlock(D)
    write = mods.WriteHead(d_model=D, tape_dim=8, zero_init=False)
    tread = mods.TapeRead(
        d_model=D, tape_dim=8, max_writes=4, n_heads=2, zero_init=False
    )

    # Capture first-iter v_k to check grad reaches it.
    h_out, tape = _wodd_forward(mods, h0, rec, write, tread, T_step=3)
    first_entry = tape._entries[0]
    first_entry.retain_grad()  # detach hook so grad is kept

    loss = h_out.sum()
    loss.backward()

    assert h0.grad is not None and torch.isfinite(h0.grad).all()
    # Even though _wodd_forward already consumed first_entry inside the
    # autograd graph, retain_grad() lets us check that grad arrives there.
    # (When unused after t=1, its grad is exactly 0 -- but the buffer
    # still exists.)
    assert first_entry.grad is not None
    assert torch.isfinite(first_entry.grad).all()


def test_append_only_across_iterations(mods):
    """After T_step iterations, tape length == T_step and all entries are
    in order (we cannot reorder or remove)."""
    torch.manual_seed(0)
    D = 16
    h0 = torch.randn(1, 3, D)
    rec = MockRecurrentBlock(D)
    write = mods.WriteHead(d_model=D, tape_dim=8, zero_init=False)
    tread = mods.TapeRead(
        d_model=D, tape_dim=8, max_writes=8, n_heads=2, zero_init=False
    )

    T = 4
    _, tape = _wodd_forward(mods, h0, rec, write, tread, T_step=T)
    assert len(tape) == T
    # No mutation API exists.
    assert not hasattr(tape, "pop")
    assert not hasattr(tape, "remove")
    assert not hasattr(tape, "__delitem__")


def test_gate_sparsity_pressure_works(mods):
    """If we add a sparsity penalty on gates, optimizer should drive the
    gate bias DOWN. Sanity check that the sparsity loss does what we want.
    """
    torch.manual_seed(0)
    D = 16
    write = mods.WriteHead(
        d_model=D, tape_dim=8, zero_init=False, gate_bias_init=0.0
    )
    opt = torch.optim.SGD(write.parameters(), lr=1.0)

    initial_bias = write.gate_proj.bias.detach().clone()
    for _ in range(20):
        pool = torch.randn(8, D)
        g, _ = write(pool)
        # Pure sparsity loss (push gate towards 0).
        loss = g.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_bias = write.gate_proj.bias.detach()
    assert (final_bias < initial_bias - 0.1).item(), (
        f"Sparsity loss should drive gate bias down. "
        f"init={float(initial_bias):.4f}, final={float(final_bias):.4f}"
    )
