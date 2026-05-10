"""Trajectory cache I/O for the ReEntry-TD distillation pipeline.

How to run a quick self-check::

    python -c "from dllm.pipelines.llada_looped import cache_io; cache_io._sanity()"

Cache directory layout::

    .cache/teacher_traj/<name>/
    ├── manifest.json
    ├── shard_0000.npz
    ├── shard_0001.npz
    └── ...

Each shard packs `shard_size` samples of teacher trajectory, with all
per-step arrays in a uniform shape `[B, num_steps, max_response_len, ...]`.

Hard rules (mirrored from REENTRY_TD_SPEC.md §13):

* bf16 fields are stored as uint16 views (numpy lacks a native bf16
  dtype). Any code that consumes them must cast to fp32 before any
  ``exp`` / ``log`` / sum.
* Mask bits are stored bit-packed via ``np.packbits``; downstream code
  unpacks back to bool with the original ``L_resp`` length.
* Manifest stamps git/sampler/torch/cuda versions; the reader refuses
  to load a manifest produced by a different commit unless the caller
  passes ``allow_stale=True``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryCacheConfig:
    """Builder-side cache config; the manifest mirrors this verbatim.

    Every field that influences sampler determinism contributes to
    ``compute_sampler_config_hash`` so the reader can detect drift.
    """

    teacher_model_path: str
    dataset_path: str
    dataset_split: str
    num_samples: int
    teacher_steps: int
    max_response_len: int
    top_k: int
    block_size: int
    shard_size: int
    seed_base: int
    mask_token_id: int
    eos_token_id: int
    bos_token_id: int
    # tokenizer fingerprint (vocab size + name) -- guards against silent
    # tokenizer changes between cache build and student training.
    vocab_size: int
    tokenizer_name_or_path: str


@dataclass
class ShardInfo:
    path: str
    sample_ids: list[int]
    t_shard: int  # max prompt_len + max_response_len in this shard
    md5: Optional[str] = None


@dataclass
class CacheManifest:
    """The on-disk manifest. Stored as JSON next to the shards."""

    config: TrajectoryCacheConfig
    git_commit_hash: str
    sampler_config_hash: str
    torch_version: str
    cuda_version: Optional[str]
    created_at: str
    shards: list[ShardInfo] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, cache_dir: str | os.PathLike) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "manifest.json"
        payload = {
            "config": dataclasses.asdict(self.config),
            "git_commit_hash": self.git_commit_hash,
            "sampler_config_hash": self.sampler_config_hash,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "created_at": self.created_at,
            "shards": [dataclasses.asdict(s) for s in self.shards],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(
        cls,
        cache_dir: str | os.PathLike,
        allow_stale: bool = False,
    ) -> "CacheManifest":
        cache_dir = Path(cache_dir)
        path = cache_dir / "manifest.json"
        with path.open("r") as f:
            payload = json.load(f)
        config = TrajectoryCacheConfig(**payload["config"])
        manifest = cls(
            config=config,
            git_commit_hash=payload["git_commit_hash"],
            sampler_config_hash=payload["sampler_config_hash"],
            torch_version=payload["torch_version"],
            cuda_version=payload.get("cuda_version"),
            created_at=payload["created_at"],
            shards=[ShardInfo(**s) for s in payload["shards"]],
        )

        if not allow_stale:
            _check_manifest_freshness(manifest)
        return manifest

    def shard_path(self, cache_dir: str | os.PathLike, shard_idx: int) -> Path:
        return Path(cache_dir) / self.shards[shard_idx].path


def _check_manifest_freshness(manifest: CacheManifest) -> None:
    """Compare manifest's git commit to current HEAD; warn loudly on drift.

    We do not raise -- training scripts should pass ``allow_stale=True``
    explicitly when they accept stale caches. The check exists so a
    silently-rebuilt cache cannot poison a downstream run.
    """
    try:
        current_commit = _git_commit_hash()
    except Exception:
        return  # not a git checkout, skip
    if current_commit and current_commit != manifest.git_commit_hash:
        raise ValueError(
            f"Cache was built at commit {manifest.git_commit_hash} but the "
            f"current HEAD is {current_commit}. Pass allow_stale=True to "
            "force load. (See REENTRY_TD_SPEC.md §13.)"
        )


# ---------------------------------------------------------------------------
# In-memory shard representation
# ---------------------------------------------------------------------------


@dataclass
class ShardData:
    """One shard, all arrays in their friendly numpy form (mask unpacked,
    bf16 fields already cast to fp32).

    Used by the cache builder to assemble a shard before write, and by
    the test roundtrip code. The training-time reader uses
    :func:`open_shard_lazy` and avoids materializing fp32 versions of the
    bulk arrays.
    """

    sample_ids: np.ndarray  # int64  [B]
    prompt_lens: np.ndarray  # int32  [B]
    final_sequences: np.ndarray  # int64  [B, T_shard]
    mask_state: np.ndarray  # bool   [B, num_steps, L_resp]
    top_k_indices: np.ndarray  # int32  [B, num_steps, L_resp, K]
    top_k_logprobs: np.ndarray  # fp32  [B, num_steps, L_resp, K]
    neg_log_tail_mass: np.ndarray  # fp32  [B, num_steps, L_resp]
    top1_conf: np.ndarray  # fp32  [B, num_steps, L_resp]
    entropy: np.ndarray  # fp32  [B, num_steps, L_resp]


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------


def pack_mask_bits(mask: np.ndarray) -> np.ndarray:
    """Bit-pack a bool mask along its last axis.

    Args:
        mask: bool array of shape [..., L_resp].
    Returns:
        uint8 array of shape [..., ceil(L_resp / 8)].
    """
    if mask.dtype != bool:
        raise TypeError(f"mask must be bool, got {mask.dtype}")
    return np.packbits(mask, axis=-1, bitorder="big")


def unpack_mask_bits(packed: np.ndarray, length: int) -> np.ndarray:
    """Inverse of :func:`pack_mask_bits`.

    Args:
        packed: uint8 array of shape [..., ceil(length / 8)].
        length: original ``L_resp``.
    Returns:
        bool array of shape [..., length].
    """
    if packed.dtype != np.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    unpacked = np.unpackbits(packed, axis=-1, count=length, bitorder="big")
    return unpacked.astype(bool)


# ---------------------------------------------------------------------------
# bf16 <-> uint16 view conversion (lazy torch import)
# ---------------------------------------------------------------------------


def torch_bf16_to_uint16_numpy(t) -> np.ndarray:
    """Convert a torch.bfloat16 tensor to a numpy uint16 array (same bits).

    Numpy has no native bf16, so we view the same memory as uint16. The
    consumer must call :func:`uint16_numpy_to_torch_bf16` before any
    arithmetic.
    """
    import torch

    if t.dtype != torch.bfloat16:
        raise TypeError(f"expected bf16 tensor, got {t.dtype}")
    return t.contiguous().view(torch.uint16).cpu().numpy()


def uint16_numpy_to_torch_bf16(arr: np.ndarray):
    """Inverse of :func:`torch_bf16_to_uint16_numpy`."""
    import torch

    if arr.dtype != np.uint16:
        raise TypeError(f"expected uint16 array, got {arr.dtype}")
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.view(torch.bfloat16)


def fp32_numpy_to_uint16_bf16(arr: np.ndarray) -> np.ndarray:
    """fp32 numpy -> uint16-view of bf16 numpy. CPU only.

    This goes through torch because numpy lacks a bf16 dtype. Acceptable
    overhead for cache writes -- happens once per shard.
    """
    import torch

    if arr.dtype != np.float32:
        raise TypeError(f"expected float32 array, got {arr.dtype}")
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)
    return t.view(torch.uint16).numpy()


def uint16_bf16_to_fp32_numpy(arr: np.ndarray) -> np.ndarray:
    """uint16-view of bf16 -> fp32 numpy. Use only at I/O boundaries."""
    return uint16_numpy_to_torch_bf16(arr).float().numpy()


# ---------------------------------------------------------------------------
# Shard write / read
# ---------------------------------------------------------------------------


_ARRAY_KEYS = (
    "sample_ids",
    "prompt_lens",
    "final_sequences",
    "mask_state_packed",
    "top_k_indices",
    "top_k_logprobs_bf16",
    "neg_log_tail_mass_bf16",
    "top1_conf_bf16",
    "entropy_bf16",
)


def write_shard(path: str | os.PathLike, shard: ShardData) -> str:
    """Serialize a :class:`ShardData` to an uncompressed npz file.

    Returns the md5 of the written file (for the manifest).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _validate_shard_shapes(shard)

    payload = {
        "sample_ids": shard.sample_ids.astype(np.int64, copy=False),
        "prompt_lens": shard.prompt_lens.astype(np.int32, copy=False),
        "final_sequences": shard.final_sequences.astype(np.int64, copy=False),
        "mask_state_packed": pack_mask_bits(shard.mask_state.astype(bool, copy=False)),
        "top_k_indices": shard.top_k_indices.astype(np.int32, copy=False),
        "top_k_logprobs_bf16": fp32_numpy_to_uint16_bf16(
            shard.top_k_logprobs.astype(np.float32, copy=False)
        ),
        "neg_log_tail_mass_bf16": fp32_numpy_to_uint16_bf16(
            shard.neg_log_tail_mass.astype(np.float32, copy=False)
        ),
        "top1_conf_bf16": fp32_numpy_to_uint16_bf16(
            shard.top1_conf.astype(np.float32, copy=False)
        ),
        "entropy_bf16": fp32_numpy_to_uint16_bf16(
            shard.entropy.astype(np.float32, copy=False)
        ),
    }

    # Uncompressed -- mmap_mode='r' requires this.
    np.savez(path, **payload)

    # Recompute file md5 for manifest integrity.
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def read_shard(
    path: str | os.PathLike,
    mmap: bool = True,
    materialize_fp32: bool = True,
) -> ShardData:
    """Read a shard back into :class:`ShardData`.

    Args:
        path: path to the .npz file.
        mmap: if True, use ``np.load(mmap_mode='r')`` for lazy access.
        materialize_fp32: if True, convert all bf16-as-uint16 fields to
            fp32 (convenient but allocates). If False, stash the uint16
            arrays directly in the corresponding fp32 fields -- callers
            doing per-step access should set this False and call
            :func:`uint16_bf16_to_fp32_numpy` per slice.
    """
    mode = "r" if mmap else None
    data = np.load(path, mmap_mode=mode, allow_pickle=False)

    mask_state_packed = np.asarray(data["mask_state_packed"])
    L_resp = data["top_k_indices"].shape[2]
    mask_state = unpack_mask_bits(mask_state_packed, length=L_resp)

    if materialize_fp32:
        top_k_logprobs = uint16_bf16_to_fp32_numpy(np.asarray(data["top_k_logprobs_bf16"]))
        neg_log_tail = uint16_bf16_to_fp32_numpy(
            np.asarray(data["neg_log_tail_mass_bf16"])
        )
        top1_conf = uint16_bf16_to_fp32_numpy(np.asarray(data["top1_conf_bf16"]))
        entropy = uint16_bf16_to_fp32_numpy(np.asarray(data["entropy_bf16"]))
    else:
        # Caller is responsible for casting per slice.
        top_k_logprobs = np.asarray(data["top_k_logprobs_bf16"])
        neg_log_tail = np.asarray(data["neg_log_tail_mass_bf16"])
        top1_conf = np.asarray(data["top1_conf_bf16"])
        entropy = np.asarray(data["entropy_bf16"])

    return ShardData(
        sample_ids=np.asarray(data["sample_ids"]),
        prompt_lens=np.asarray(data["prompt_lens"]),
        final_sequences=np.asarray(data["final_sequences"]),
        mask_state=mask_state,
        top_k_indices=np.asarray(data["top_k_indices"]),
        top_k_logprobs=top_k_logprobs,
        neg_log_tail_mass=neg_log_tail,
        top1_conf=top1_conf,
        entropy=entropy,
    )


def _validate_shard_shapes(shard: ShardData) -> None:
    B = shard.sample_ids.shape[0]
    assert shard.prompt_lens.shape == (B,)
    assert shard.final_sequences.ndim == 2 and shard.final_sequences.shape[0] == B
    assert shard.mask_state.ndim == 3 and shard.mask_state.shape[0] == B
    num_steps = shard.mask_state.shape[1]
    L_resp = shard.mask_state.shape[2]
    K = shard.top_k_indices.shape[-1]
    expected_top_shape = (B, num_steps, L_resp, K)
    expected_per_pos = (B, num_steps, L_resp)
    assert shard.top_k_indices.shape == expected_top_shape, shard.top_k_indices.shape
    assert shard.top_k_logprobs.shape == expected_top_shape, shard.top_k_logprobs.shape
    assert shard.neg_log_tail_mass.shape == expected_per_pos, shard.neg_log_tail_mass.shape
    assert shard.top1_conf.shape == expected_per_pos, shard.top1_conf.shape
    assert shard.entropy.shape == expected_per_pos, shard.entropy.shape


# ---------------------------------------------------------------------------
# Per-step input reconstruction (used by training collator)
# ---------------------------------------------------------------------------


def reconstruct_input_at_step(
    final_sequence: np.ndarray,
    prompt_len: int,
    mask_at_step: np.ndarray,
    mask_token_id: int,
    response_len: int | None = None,
) -> np.ndarray:
    """Build the noised input that the teacher saw at the start of a step.

    Args:
        final_sequence: int64 [T] -- the fully-decoded canvas.
        prompt_len: number of prompt tokens (response begins at this index).
        mask_at_step: bool [L_resp] -- True where this step had a mask.
        mask_token_id: id of the mask token.
        response_len: defaults to ``mask_at_step.shape[0]``; can be smaller
            if you want to slice off trailing padding (the model still
            sees those positions, just with EOS, so the default is fine
            in practice).

    Returns:
        int64 [T] -- the input the student should see for this step.
    """
    if response_len is None:
        response_len = mask_at_step.shape[0]
    out = final_sequence.copy()
    response_slice = slice(prompt_len, prompt_len + response_len)
    response = out[response_slice]
    response = np.where(mask_at_step[:response_len], mask_token_id, response)
    out[response_slice] = response
    return out


# ---------------------------------------------------------------------------
# Sampler config hashing + git commit
# ---------------------------------------------------------------------------


def compute_sampler_config_hash(payload: dict[str, Any]) -> str:
    """Hash sampler-relevant config so manifest readers can detect drift.

    Pass in only the fields that affect sampler determinism (e.g.
    ``teacher_steps``, ``top_k``, ``block_size``, ``temperature``,
    ``stochastic_transfer``, ``seed_base``). The hash is used as the
    cache's reproducibility stamp.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _git_commit_hash() -> Optional[str]:
    """Resolve the current git HEAD short hash, or None if not a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _sanity() -> None:
    """Tiny round-trip check; runnable without a GPU.

    Builds a fake 2-sample shard, writes, reads, asserts byte/pattern
    equality on bool/int fields and bf16-precision equality on numerical
    fields.
    """
    rng = np.random.default_rng(0)
    B, num_steps, L_resp, K = 2, 4, 8, 16
    T_shard = 10 + L_resp

    shard = ShardData(
        sample_ids=np.array([1, 2], dtype=np.int64),
        prompt_lens=np.array([10, 7], dtype=np.int32),
        final_sequences=rng.integers(0, 1000, size=(B, T_shard), dtype=np.int64),
        mask_state=rng.integers(0, 2, size=(B, num_steps, L_resp), dtype=np.int8).astype(
            bool
        ),
        top_k_indices=rng.integers(0, 1000, size=(B, num_steps, L_resp, K), dtype=np.int32),
        top_k_logprobs=rng.standard_normal((B, num_steps, L_resp, K)).astype(np.float32),
        neg_log_tail_mass=rng.uniform(0, 30, size=(B, num_steps, L_resp)).astype(np.float32),
        top1_conf=rng.uniform(0, 1, size=(B, num_steps, L_resp)).astype(np.float32),
        entropy=rng.uniform(0, 8, size=(B, num_steps, L_resp)).astype(np.float32),
    )

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "shard.npz")
        write_shard(path, shard)
        round_tripped = read_shard(path, mmap=False, materialize_fp32=True)

    # Exact equality for int / bool fields.
    assert np.array_equal(shard.sample_ids, round_tripped.sample_ids)
    assert np.array_equal(shard.prompt_lens, round_tripped.prompt_lens)
    assert np.array_equal(shard.final_sequences, round_tripped.final_sequences)
    assert np.array_equal(shard.mask_state, round_tripped.mask_state)
    assert np.array_equal(shard.top_k_indices, round_tripped.top_k_indices)

    # bf16 precision -- typical relative error is < 1/256 = 0.39%.
    for name in ("top_k_logprobs", "neg_log_tail_mass", "top1_conf", "entropy"):
        a = getattr(shard, name)
        b = getattr(round_tripped, name)
        # Compare in fp32. bf16 has 7-bit mantissa, so worst-case relative
        # error ~ 2^-7 ~ 0.78%. Use 1% tolerance with a 1e-3 floor.
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-3, err_msg=name)

    print("cache_io._sanity OK: shard round-trip preserves all fields within bf16 precision")


if __name__ == "__main__":
    _sanity()
