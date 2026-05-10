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
import warnings

import numpy as np
import pytest

from dllm.pipelines.llada_looped.cache_io import (
    CacheManifest,
    ShardData,
    ShardInfo,
    StreamingShardWriter,
    TrajectoryCacheConfig,
    compute_sampler_config_hash,
    fp32_numpy_to_uint16_bf16,
    open_shard_arrays,
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
    assert packed.shape == (4, 7, 8)
    unpacked = unpack_mask_bits(packed, length=64)
    np.testing.assert_array_equal(mask, unpacked)


def test_mask_bit_roundtrip_unaligned():
    """L_resp not a multiple of 8 -- packbits pads, unpack must trim."""
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, size=(2, 3, 13)).astype(bool)
    packed = pack_mask_bits(mask)
    assert packed.shape == (2, 3, 2)
    unpacked = unpack_mask_bits(packed, length=13)
    assert unpacked.shape == (2, 3, 13)
    np.testing.assert_array_equal(mask, unpacked)


# ---------------------------------------------------------------------------
# bf16 view
# ---------------------------------------------------------------------------


def test_bf16_view_roundtrip_precision():
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
    np.testing.assert_allclose(a, b, rtol=1e-2)


# ---------------------------------------------------------------------------
# Shard write / read / mmap
# ---------------------------------------------------------------------------


def test_full_shard_roundtrip():
    """Write -> read returns same content within bf16 precision."""
    shard = _make_shard()
    with tempfile.TemporaryDirectory() as td:
        shard_dir = os.path.join(td, "shard_0000")
        write_shard(shard_dir, shard)
        rt = read_shard(shard_dir, mmap=False, materialize_fp32=True)

    np.testing.assert_array_equal(shard.sample_ids, rt.sample_ids)
    np.testing.assert_array_equal(shard.prompt_lens, rt.prompt_lens)
    np.testing.assert_array_equal(shard.final_sequences, rt.final_sequences)
    np.testing.assert_array_equal(shard.mask_state, rt.mask_state)
    np.testing.assert_array_equal(shard.top_k_indices, rt.top_k_indices)
    for name in ("top_k_logprobs", "neg_log_tail_mass", "top1_conf", "entropy"):
        a = getattr(shard, name)
        b = getattr(rt, name)
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-3, err_msg=name)


def test_shard_writes_per_key_npy_files():
    """Sanity: write_shard produces a directory of .npy files (not a .npz)."""
    shard = _make_shard()
    with tempfile.TemporaryDirectory() as td:
        shard_dir = os.path.join(td, "shard_0000")
        write_shard(shard_dir, shard)
        files = sorted(os.listdir(shard_dir))
        # one .npy per array key
        assert all(f.endswith(".npy") for f in files), files
        for required in ("sample_ids.npy", "top_k_logprobs_bf16.npy",
                          "mask_state_packed.npy"):
            assert required in files, f"missing {required}"


def test_open_shard_arrays_returns_real_memmaps():
    """The training-time hot path requires real np.memmap, not full reads."""
    shard = _make_shard(B=4, num_steps=8, L_resp=16, K=8)
    with tempfile.TemporaryDirectory() as td:
        shard_dir = os.path.join(td, "shard_0000")
        write_shard(shard_dir, shard)

        handle = open_shard_arrays(shard_dir)
        for key in handle:
            arr = handle[key]
            # np.memmap is a subclass of np.ndarray; its presence proves we
            # didn't fall back to a full eager read.
            assert isinstance(arr, np.memmap), (
                f"{key} is not memory-mapped (type={type(arr).__name__})"
            )

        # Per-step slicing should preserve dtype + shape without forcing a
        # full read. We can't easily measure I/O, but shape correctness +
        # data correctness is a proxy.
        slice_uint16 = handle["top_k_logprobs_bf16"][0, 0]
        assert slice_uint16.shape == (16, 8)
        assert slice_uint16.dtype == np.uint16


def test_shard_validates_shapes():
    shard = _make_shard()
    shard.top_k_indices = shard.top_k_indices[..., :5]  # break K
    with tempfile.TemporaryDirectory() as td, pytest.raises(AssertionError):
        write_shard(os.path.join(td, "shard_0000"), shard)


# ---------------------------------------------------------------------------
# StreamingShardWriter parity
# ---------------------------------------------------------------------------


def test_streaming_writer_matches_eager_write():
    """``StreamingShardWriter`` and ``write_shard(ShardData)`` should
    produce caches that read back identically up to bf16 precision.

    The two paths are intended to be equivalent: eager builds a buffered
    ShardData and dumps; streaming writes per-(sample, step). The
    streaming writer is what the production builder uses to keep peak
    RAM at one micro-batch's worth instead of the whole shard's fp32
    buffer.
    """
    shard = _make_shard(B=3, num_steps=4, L_resp=8, K=8)
    B = shard.sample_ids.shape[0]
    num_steps = shard.mask_state.shape[1]
    L_resp = shard.mask_state.shape[2]
    K = shard.top_k_indices.shape[-1]
    t_shard = shard.final_sequences.shape[1]

    with tempfile.TemporaryDirectory() as td:
        eager_dir = os.path.join(td, "eager")
        write_shard(eager_dir, shard)

        stream_dir = os.path.join(td, "stream")
        with StreamingShardWriter(
            stream_dir, B=B, t_shard=t_shard, num_steps=num_steps,
            L_resp=L_resp, K=K, final_sequences_pad=0,
        ) as w:
            for i in range(B):
                w.set_sample(
                    i,
                    sample_id=int(shard.sample_ids[i]),
                    prompt_len=int(shard.prompt_lens[i]),
                    final_sequence=shard.final_sequences[i],
                )
                for step_idx in range(num_steps):
                    w.set_step(
                        i, step_idx,
                        mask_state=shard.mask_state[i, step_idx],
                        top_k_indices=shard.top_k_indices[i, step_idx],
                        top_k_logprobs=shard.top_k_logprobs[i, step_idx],
                        neg_log_tail_mass=shard.neg_log_tail_mass[i, step_idx],
                        top1_conf=shard.top1_conf[i, step_idx],
                        entropy=shard.entropy[i, step_idx],
                    )

        eager = read_shard(eager_dir, mmap=False, materialize_fp32=True)
        stream = read_shard(stream_dir, mmap=False, materialize_fp32=True)

    np.testing.assert_array_equal(eager.sample_ids, stream.sample_ids)
    np.testing.assert_array_equal(eager.prompt_lens, stream.prompt_lens)
    np.testing.assert_array_equal(eager.final_sequences, stream.final_sequences)
    np.testing.assert_array_equal(eager.mask_state, stream.mask_state)
    np.testing.assert_array_equal(eager.top_k_indices, stream.top_k_indices)
    # bf16 fields go through identical fp32->uint16->fp32 paths so they
    # should match bit-for-bit.
    for name in ("top_k_logprobs", "neg_log_tail_mass", "top1_conf", "entropy"):
        np.testing.assert_array_equal(
            getattr(eager, name), getattr(stream, name), err_msg=name
        )


def test_streaming_writer_pads_final_sequences_with_eos():
    """``final_sequences_pad`` initializes the trailing per-sample columns
    so a partial set_sample (e.g. when a sample's prompt is shorter than
    the shard's max) leaves EOS, not zeros, in the gap.
    """
    B = 2
    t_shard = 10
    num_steps = 2
    L_resp = 4
    K = 4
    eos_id = 999

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "shard")
        with StreamingShardWriter(
            path, B=B, t_shard=t_shard, num_steps=num_steps,
            L_resp=L_resp, K=K, final_sequences_pad=eos_id,
        ) as w:
            # Sample 0: full row.
            w.set_sample(
                0, sample_id=0, prompt_len=3,
                final_sequence=np.arange(t_shard, dtype=np.int64),
            )
            # Sample 1: partial row (only first 5 positions).
            w.set_sample(
                1, sample_id=1, prompt_len=2,
                final_sequence=np.array([10, 11, 12, 13, 14], dtype=np.int64),
            )
            for i in range(B):
                for s in range(num_steps):
                    w.set_step(
                        i, s,
                        mask_state=np.zeros(L_resp, dtype=bool),
                        top_k_indices=np.zeros((L_resp, K), dtype=np.int32),
                        top_k_logprobs=np.zeros((L_resp, K), dtype=np.float32),
                        neg_log_tail_mass=np.zeros(L_resp, dtype=np.float32),
                        top1_conf=np.zeros(L_resp, dtype=np.float32),
                        entropy=np.zeros(L_resp, dtype=np.float32),
                    )

        rt = read_shard(path, mmap=False, materialize_fp32=True)

    assert rt.final_sequences[0].tolist() == list(range(t_shard))
    assert rt.final_sequences[1].tolist() == [10, 11, 12, 13, 14] + [eos_id] * 5


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
            ShardInfo(path="shard_0000", sample_ids=[0, 1, 2], t_shard=300),
            ShardInfo(path="shard_0001", sample_ids=[3, 4, 5], t_shard=280),
        ],
    )

    with tempfile.TemporaryDirectory() as td:
        manifest.save(td)
        # Synthetic git hash; default load must warn (not raise) and succeed.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = CacheManifest.load(td)
        # The synthetic abc123 won't match current HEAD, so we expect a warning.
        # (If we happen to be in a git-less checkout, no warning is fine.)
        # We don't assert on count; just that loading itself succeeded.

    assert loaded.git_commit_hash == manifest.git_commit_hash
    assert loaded.sampler_config_hash == manifest.sampler_config_hash
    assert loaded.config == manifest.config
    assert len(loaded.shards) == len(manifest.shards)
    for a, b in zip(loaded.shards, manifest.shards):
        assert a.path == b.path
        assert a.sample_ids == b.sample_ids
        assert a.t_shard == b.t_shard


def test_manifest_strict_git_raises_on_drift():
    """strict_git=True must raise if HEAD differs from manifest's commit."""
    cfg = _make_config()
    manifest = CacheManifest(
        config=cfg,
        git_commit_hash="0" * 40,  # not the current HEAD
        sampler_config_hash="dead",
        torch_version="2.5.0",
        cuda_version=None,
        created_at=utcnow_iso(),
        shards=[],
    )
    with tempfile.TemporaryDirectory() as td:
        manifest.save(td)
        # Default: warn only.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = CacheManifest.load(td)
            assert loaded.git_commit_hash == "0" * 40

        # strict_git=True: raise.
        with pytest.raises(ValueError, match="commit"):
            CacheManifest.load(td, strict_git=True)


# ---------------------------------------------------------------------------
# Sampler config hash
# ---------------------------------------------------------------------------


def test_sampler_config_hash_stable():
    h1 = compute_sampler_config_hash({"a": 1, "b": 2.0})
    h2 = compute_sampler_config_hash({"b": 2.0, "a": 1})
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
