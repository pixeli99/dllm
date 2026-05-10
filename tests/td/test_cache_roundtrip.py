"""Cache I/O roundtrip tests.

How to run::

    cd /Users/pixeli/dllm
    python -m pytest tests/td/test_cache_roundtrip.py -v

These tests don't need a GPU or a real model -- they only exercise
:mod:`dllm.pipelines.llada_looped.cache_io`.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from dllm.pipelines.llada_looped.cache_io import (
    CacheManifest,
    ShardData,
    ShardInfo,
    TrajectoryCacheConfig,
    compute_sampler_config_hash,
    fp32_numpy_to_uint16_bf16,
    pack_mask_bits,
    read_shard,
    uint16_bf16_to_fp32_numpy,
    unpack_mask_bits,
    utcnow_iso,
    write_shard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shard(B: int = 3, num_steps: int = 5, L_resp: int = 8, K: int = 16,
                seed: int = 0, T_shard: int | None = None) -> ShardData:
    if T_shard is None:
        T_shard = 12 + L_resp
    rng = np.random.default_rng(seed)
    return ShardData(
        sample_ids=np.arange(B, dtype=np.int64),
        prompt_lens=np.array([12, 9, 7][:B], dtype=np.int32) if B <= 3
            else rng.integers(4, 12, size=B, dtype=np.int32),
        final_sequences=rng.integers(0, 1000, size=(B, T_shard), dtype=np.int64),
        mask_state=rng.integers(0, 2, size=(B, num_steps, L_resp)).astype(bool),
        top_k_indices=rng.integers(0, 1000, size=(B, num_steps, L_resp, K), dtype=np.int32),
        # Realistic value ranges for log-prob style fields
        top_k_logprobs=rng.uniform(-30, 0, size=(B, num_steps, L_resp, K)).astype(np.float32),
        neg_log_tail_mass=rng.uniform(0, 50, size=(B, num_steps, L_resp)).astype(np.float32),
        top1_conf=rng.uniform(0, 1, size=(B, num_steps, L_resp)).astype(np.float32),
        entropy=rng.uniform(0, 8, size=(B, num_steps, L_resp)).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Bit-pack / unpack
# ---------------------------------------------------------------------------


def test_mask_bit_roundtrip_aligned():
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, size=(4, 7, 64)).astype(bool)
    packed = pack_mask_bits(mask)
    assert packed.dtype == np.uint8
    assert packed.shape == (4, 7, 8)  # 64 / 8
    unpacked = unpack_mask_bits(packed, length=64)
    np.testing.assert_array_equal(mask, unpacked)


def test_mask_bit_roundtrip_unaligned():
    """L_resp not a multiple of 8 -- packbits pads, unpack must trim."""
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, size=(2, 3, 13)).astype(bool)
    packed = pack_mask_bits(mask)
    assert packed.shape == (2, 3, 2)  # ceil(13/8) = 2
    unpacked = unpack_mask_bits(packed, length=13)
    assert unpacked.shape == (2, 3, 13)
    np.testing.assert_array_equal(mask, unpacked)


# ---------------------------------------------------------------------------
# bf16 view
# ---------------------------------------------------------------------------


def test_bf16_view_roundtrip_precision():
    """bf16 has 7-bit mantissa -> ~0.78% worst-case relative error."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-30, 30, size=(64, 64)).astype(np.float32)
    packed = fp32_numpy_to_uint16_bf16(a)
    assert packed.dtype == np.uint16
    assert packed.shape == a.shape
    b = uint16_bf16_to_fp32_numpy(packed)
    np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-3)


def test_bf16_view_preserves_zeros():
    a = np.zeros((4, 4), dtype=np.float32)
    b = uint16_bf16_to_fp32_numpy(fp32_numpy_to_uint16_bf16(a))
    assert np.array_equal(a, b)


def test_bf16_view_preserves_special_values():
    a = np.array([0.0, -0.0, 1.0, -1.0, 1e-30, 1e30], dtype=np.float32)
    b = uint16_bf16_to_fp32_numpy(fp32_numpy_to_uint16_bf16(a))
    # bf16 can represent both 1e-30 (subnormal-ish) and 1e30 (within range).
    np.testing.assert_allclose(a, b, rtol=1e-2)


# ---------------------------------------------------------------------------
# Full shard roundtrip
# ---------------------------------------------------------------------------


def test_full_shard_roundtrip():
    shard = _make_shard()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "shard.npz")
        md5 = write_shard(path, shard)
        assert isinstance(md5, str) and len(md5) == 32  # md5 hex

        rt = read_shard(path, mmap=False, materialize_fp32=True)

    # Exact equality on int / bool fields
    np.testing.assert_array_equal(shard.sample_ids, rt.sample_ids)
    np.testing.assert_array_equal(shard.prompt_lens, rt.prompt_lens)
    np.testing.assert_array_equal(shard.final_sequences, rt.final_sequences)
    np.testing.assert_array_equal(shard.mask_state, rt.mask_state)
    np.testing.assert_array_equal(shard.top_k_indices, rt.top_k_indices)

    # bf16 precision on numerical fields
    for name in ("top_k_logprobs", "neg_log_tail_mass", "top1_conf", "entropy"):
        a = getattr(shard, name)
        b = getattr(rt, name)
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-3, err_msg=name)


def test_shard_md5_changes_with_content():
    shard1 = _make_shard(seed=0)
    shard2 = _make_shard(seed=1)
    with tempfile.TemporaryDirectory() as td:
        m1 = write_shard(os.path.join(td, "a.npz"), shard1)
        m2 = write_shard(os.path.join(td, "b.npz"), shard2)
    assert m1 != m2


def test_shard_validates_shapes():
    shard = _make_shard()
    # Break a shape: top_k_indices has wrong K
    shard.top_k_indices = shard.top_k_indices[..., :5]  # K=16 -> 5
    with tempfile.TemporaryDirectory() as td, pytest.raises(AssertionError):
        write_shard(os.path.join(td, "broken.npz"), shard)


# ---------------------------------------------------------------------------
# Manifest save / load
# ---------------------------------------------------------------------------


def _make_config() -> TrajectoryCacheConfig:
    return TrajectoryCacheConfig(
        teacher_model_path="/path/to/teacher",
        dataset_path="/path/to/data",
        dataset_split="train",
        num_samples=100,
        teacher_steps=256,
        max_response_len=256,
        top_k=64,
        block_size=256,
        shard_size=50,
        seed_base=20260510,
        mask_token_id=126336,
        eos_token_id=126348,
        bos_token_id=126080,
        vocab_size=128000,
        tokenizer_name_or_path="/path/to/teacher",
    )


def test_manifest_save_load():
    cfg = _make_config()
    sampler_hash = compute_sampler_config_hash({"foo": 1, "bar": 2})
    manifest = CacheManifest(
        config=cfg,
        git_commit_hash="abc123",
        sampler_config_hash=sampler_hash,
        torch_version="2.5.0",
        cuda_version="12.4",
        created_at=utcnow_iso(),
        shards=[
            ShardInfo(path="shard_0000.npz", sample_ids=[0, 1, 2], t_shard=300, md5="d" * 32),
            ShardInfo(path="shard_0001.npz", sample_ids=[3, 4, 5], t_shard=280, md5="e" * 32),
        ],
    )

    with tempfile.TemporaryDirectory() as td:
        manifest.save(td)
        # Use allow_stale=True since we synthetic git_commit_hash != current HEAD
        loaded = CacheManifest.load(td, allow_stale=True)

    assert loaded.git_commit_hash == manifest.git_commit_hash
    assert loaded.sampler_config_hash == manifest.sampler_config_hash
    assert loaded.config == manifest.config
    assert len(loaded.shards) == len(manifest.shards)
    for a, b in zip(loaded.shards, manifest.shards):
        assert a.path == b.path
        assert a.sample_ids == b.sample_ids
        assert a.t_shard == b.t_shard


def test_manifest_stale_check_raises():
    """Manifest loaded with a fake git hash and allow_stale=False must raise."""
    cfg = _make_config()
    manifest = CacheManifest(
        config=cfg,
        git_commit_hash="0" * 40,  # definitely not current HEAD
        sampler_config_hash="dead",
        torch_version="2.5.0",
        cuda_version=None,
        created_at=utcnow_iso(),
        shards=[],
    )
    with tempfile.TemporaryDirectory() as td:
        manifest.save(td)
        # Without allow_stale, must raise (we are inside a git checkout)
        with pytest.raises(ValueError, match="commit"):
            CacheManifest.load(td, allow_stale=False)
        # With allow_stale, must succeed
        loaded = CacheManifest.load(td, allow_stale=True)
        assert loaded.git_commit_hash == "0" * 40


# ---------------------------------------------------------------------------
# Sampler config hash
# ---------------------------------------------------------------------------


def test_sampler_config_hash_stable():
    h1 = compute_sampler_config_hash({"a": 1, "b": 2.0})
    h2 = compute_sampler_config_hash({"b": 2.0, "a": 1})  # same dict, different order
    assert h1 == h2


def test_sampler_config_hash_changes_with_value():
    h1 = compute_sampler_config_hash({"a": 1})
    h2 = compute_sampler_config_hash({"a": 2})
    assert h1 != h2


# ---------------------------------------------------------------------------
# Reconstruct input at step
# ---------------------------------------------------------------------------


def test_reconstruct_input_at_step():
    from dllm.pipelines.llada_looped.cache_io import reconstruct_input_at_step

    final_seq = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    prompt_len = 3
    L_resp = 5
    mask_at_step = np.array([True, False, True, False, True])
    mask_id = 999

    out = reconstruct_input_at_step(
        final_sequence=final_seq,
        prompt_len=prompt_len,
        mask_at_step=mask_at_step,
        mask_token_id=mask_id,
    )
    expected = np.array([10, 20, 30, 999, 50, 999, 70, 999], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)
