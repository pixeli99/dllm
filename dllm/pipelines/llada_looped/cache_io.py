"""Trajectory cache I/O for the ReEntry-TD distillation pipeline.

How to run a quick self-check::

    python -c "from dllm.pipelines.llada_looped import cache_io; cache_io._sanity()"

Cache directory layout::

    .cache/teacher_traj/<name>/
    ├── manifest.json
    ├── shard_0000/
    │   ├── sample_ids.npy
    │   ├── prompt_lens.npy
    │   ├── final_sequences.npy
    │   ├── mask_state_packed.npy
    │   ├── top_k_indices.npy
    │   ├── top_k_logprobs_bf16.npy
    │   ├── neg_log_tail_mass_bf16.npy
    │   ├── top1_conf_bf16.npy
    │   └── entropy_bf16.npy
    ├── shard_0001/
    │   └── ...
    └── ...

Why one ``.npy`` per array (not a single ``.npz``):

* ``np.load(path.npz, mmap_mode='r')`` returns an ``NpzFile`` whose
  ``__getitem__`` decompresses the *whole* array into memory the first
  time you touch it. With per-step random access during training, the
  ``top_k_logprobs_bf16`` array alone is ~800 MB / shard -- that's a
  full read of the file on every cold lookup. ``np.load(path.npy,
  mmap_mode='r')`` *does* return an ``np.memmap``, which is what we
  want for random ``[sample_in_shard, step]`` slicing.

Hard rules (mirrored from REENTRY_TD_SPEC.md §13):

* bf16 fields are stored as uint16 views (numpy lacks a native bf16
  dtype). Any code that consumes them must cast to fp32 before any
  ``exp`` / ``log`` / sum. ``read_shard`` does this when
  ``materialize_fp32=True``; ``open_shard_arrays`` does NOT, leaving
  the cast for the per-step slice in the training collator.
* Mask bits are stored bit-packed via ``np.packbits``; downstream code
  unpacks back to bool with the original ``L_resp`` length.
* The manifest stamps git commit, sampler config hash, torch / cuda
  versions. Cache content is determined by ``sampler_config_hash``;
  git commit is stored only as audit metadata. ``CacheManifest.load``
  emits a warning on git drift but does not raise (pass
  ``strict_git=True`` to promote it). This is so a freshly-committed
  trainer can still consume a cache built last week.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import subprocess
import warnings
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
    mask_token_id: int
    eos_token_id: int
    bos_token_id: int
    vocab_size: int
    tokenizer_name_or_path: str


@dataclass
class ShardInfo:
    """Per-shard metadata stored in the manifest.

    ``path`` is a directory name relative to the cache root; the
    directory holds one ``.npy`` per array key.
    """

    path: str
    sample_ids: list[int]
    t_shard: int  # max prompt_len + max_response_len in this shard


@dataclass
class CacheManifest:
    """The on-disk manifest, stored as ``manifest.json`` next to the shards."""

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
        strict_git: bool = False,
    ) -> "CacheManifest":
        """Load a manifest. Warns on git drift; passes ``strict_git=True`` to
        promote that to an error.

        Sampler-config / content compatibility is *not* checked here -- the
        manifest's ``sampler_config_hash`` is recorded for audit. Callers
        that want hard config compatibility should compare it themselves.
        """
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

        _check_git_drift(manifest, strict=strict_git)
        return manifest

    def shard_dir(self, cache_dir: str | os.PathLike, shard_idx: int) -> Path:
        return Path(cache_dir) / self.shards[shard_idx].path


def _check_git_drift(manifest: CacheManifest, strict: bool) -> None:
    """Warn (or raise, if strict) when the current HEAD differs from the
    commit that built the cache.

    Cache content is not bit-tied to git -- it is determined by the
    sampler config + dataset + teacher checkpoint. The git stamp is
    audit metadata. Promoting drift to an error is opt-in.
    """
    try:
        current_commit = _git_commit_hash()
    except Exception:
        return
    if current_commit and current_commit != manifest.git_commit_hash:
        msg = (
            f"Cache was built at git commit {manifest.git_commit_hash} but "
            f"the current HEAD is {current_commit}. Cache content is "
            "determined by the sampler config hash "
            f"({manifest.sampler_config_hash}); git drift is informational."
        )
        if strict:
            raise ValueError(msg + " (strict_git=True; pass False to demote.)")
        warnings.warn(msg + " Pass strict_git=True to promote to an error.",
                      stacklevel=3)


# ---------------------------------------------------------------------------
# In-memory shard representation
# ---------------------------------------------------------------------------


@dataclass
class ShardData:
    """One shard, all arrays in their friendly numpy form.

    Used by the cache builder to assemble a shard before write, and by
    the test roundtrip code. Training-time random access goes through
    :func:`open_shard_arrays` instead, to keep the bulk arrays as
    memory-maps.
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
    """Inverse of :func:`pack_mask_bits`."""
    if packed.dtype != np.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    unpacked = np.unpackbits(packed, axis=-1, count=length, bitorder="big")
    return unpacked.astype(bool)


# ---------------------------------------------------------------------------
# bf16 <-> uint16 view conversion (lazy torch import)
# ---------------------------------------------------------------------------


def torch_bf16_to_uint16_numpy(t) -> np.ndarray:
    """Convert a torch.bfloat16 tensor to a numpy uint16 array (same bits)."""
    import torch

    if t.dtype != torch.bfloat16:
        raise TypeError(f"expected bf16 tensor, got {t.dtype}")
    return t.contiguous().view(torch.uint16).cpu().numpy()


def uint16_numpy_to_torch_bf16(arr: np.ndarray):
    """Inverse of :func:`torch_bf16_to_uint16_numpy`.

    Forces a writeable contiguous copy because ``torch.from_numpy`` warns
    on non-writable buffers (which is what mmap'd ``.npy`` files give us).
    The copy is small -- typically a per-step slice of a few KB.
    """
    import torch

    if arr.dtype != np.uint16:
        raise TypeError(f"expected uint16 array, got {arr.dtype}")
    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    t = torch.from_numpy(arr)
    return t.view(torch.bfloat16)


def fp32_numpy_to_uint16_bf16(arr: np.ndarray) -> np.ndarray:
    """fp32 numpy -> uint16-view of bf16 numpy. CPU only."""
    import torch

    if arr.dtype != np.float32:
        raise TypeError(f"expected float32 array, got {arr.dtype}")
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)
    return t.view(torch.uint16).numpy()


def uint16_bf16_to_fp32_numpy(arr: np.ndarray) -> np.ndarray:
    """uint16-view of bf16 -> fp32 numpy. Use only at I/O boundaries."""
    return uint16_numpy_to_torch_bf16(arr).float().numpy()


# ---------------------------------------------------------------------------
# Shard write / read / mmap-open
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


def write_shard(path: str | os.PathLike, shard: ShardData) -> None:
    """Serialize a :class:`ShardData` as a directory of per-key ``.npy`` files.

    Why a directory: we need real ``np.memmap`` random access during
    training. ``np.savez`` packs everything into a zip and forces a full
    read on first lookup; that's not viable for an 800 MB
    ``top_k_logprobs_bf16`` array touched once per training example.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    _validate_shard_shapes(shard)

    arrays = {
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

    for name, arr in arrays.items():
        np.save(path / f"{name}.npy", arr, allow_pickle=False)


def read_shard(
    path: str | os.PathLike,
    mmap: bool = True,
    materialize_fp32: bool = True,
) -> ShardData:
    """Read a shard back into a fully-materialized :class:`ShardData`.

    Use this for tests / audits. For training, prefer
    :func:`open_shard_arrays` which keeps every array as a memory-map
    and lets the collator slice + cast per ``(sample, step)``.

    Args:
        path: shard directory.
        mmap: if True, the underlying ``.npy`` files are opened as
            memory-maps; the returned arrays are still
            ``np.ndarray``-shaped views. Set False to force eager loads.
        materialize_fp32: convert the bf16-as-uint16 fields to fp32.
    """
    path = Path(path)
    mode = "r" if mmap else None

    def _load(name: str) -> np.ndarray:
        return np.load(path / f"{name}.npy", mmap_mode=mode, allow_pickle=False)

    sample_ids = np.asarray(_load("sample_ids"))
    prompt_lens = np.asarray(_load("prompt_lens"))
    final_sequences = np.asarray(_load("final_sequences"))
    mask_state_packed = np.asarray(_load("mask_state_packed"))
    top_k_indices = np.asarray(_load("top_k_indices"))

    L_resp = int(top_k_indices.shape[2])
    mask_state = unpack_mask_bits(mask_state_packed, length=L_resp)

    if materialize_fp32:
        top_k_logprobs = uint16_bf16_to_fp32_numpy(np.asarray(_load("top_k_logprobs_bf16")))
        neg_log_tail = uint16_bf16_to_fp32_numpy(
            np.asarray(_load("neg_log_tail_mass_bf16"))
        )
        top1_conf = uint16_bf16_to_fp32_numpy(np.asarray(_load("top1_conf_bf16")))
        entropy = uint16_bf16_to_fp32_numpy(np.asarray(_load("entropy_bf16")))
    else:
        top_k_logprobs = np.asarray(_load("top_k_logprobs_bf16"))
        neg_log_tail = np.asarray(_load("neg_log_tail_mass_bf16"))
        top1_conf = np.asarray(_load("top1_conf_bf16"))
        entropy = np.asarray(_load("entropy_bf16"))

    return ShardData(
        sample_ids=sample_ids,
        prompt_lens=prompt_lens,
        final_sequences=final_sequences,
        mask_state=mask_state,
        top_k_indices=top_k_indices,
        top_k_logprobs=top_k_logprobs,
        neg_log_tail_mass=neg_log_tail,
        top1_conf=top1_conf,
        entropy=entropy,
    )


def open_shard_arrays(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Return a dict of memory-mapped arrays for one shard.

    The training collator should hold one of these per shard and slice
    by ``[sample_in_shard, step]`` to read just the needed pages. bf16
    fields are returned as uint16 views; the collator must call
    :func:`uint16_bf16_to_fp32_numpy` (or the per-element equivalent)
    before any arithmetic.
    """
    path = Path(path)
    out: dict[str, np.ndarray] = {}
    for key in _ARRAY_KEYS:
        out[key] = np.load(path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
    return out


# ---------------------------------------------------------------------------
# Streaming shard writer (production builds; eager write_shard is for tests)
# ---------------------------------------------------------------------------


class StreamingShardWriter:
    """Stream a shard directory one ``(sample, step)`` at a time, to
    memory-mapped ``.npy`` files.

    The eager :func:`write_shard` allocates the whole shard's fp32
    ``top_k_logprobs`` array (~1.6 GB at shard_size=100, L_resp=256, K=64)
    in RAM before converting to bf16 and dumping. For real cache builds
    the peak is unnecessary -- we know the per-step shape upfront, so we
    can pre-allocate memmap'd ``.npy`` files and write each ``(sample,
    step)`` slice directly. RAM cost drops to ~per-micro-batch.

    Use as a context manager::

        with StreamingShardWriter(path, B=100, t_shard=512, num_steps=256,
                                  L_resp=256, K=64,
                                  final_sequences_pad=eos_id) as w:
            for i in range(B):
                w.set_sample(i, sample_id=..., prompt_len=..., final_sequence=...)
                for step_idx in range(num_steps):
                    w.set_step(i, step_idx, mask_state=..., top_k_indices=...,
                                top_k_logprobs=..., neg_log_tail_mass=...,
                                top1_conf=..., entropy=...)
    """

    _ARRAY_NAMES = (
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

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        B: int,
        t_shard: int,
        num_steps: int,
        L_resp: int,
        K: int,
        final_sequences_pad: int = 0,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.B = B
        self.num_steps = num_steps
        self.L_resp = L_resp
        self.K = K
        mask_packed_len = (L_resp + 7) // 8

        def _mm(name: str, shape: tuple[int, ...], dtype) -> np.memmap:
            return np.lib.format.open_memmap(
                self.path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
            )

        self.sample_ids = _mm("sample_ids", (B,), np.int64)
        self.prompt_lens = _mm("prompt_lens", (B,), np.int32)
        self.final_sequences = _mm("final_sequences", (B, t_shard), np.int64)
        if final_sequences_pad != 0:
            self.final_sequences[:] = final_sequences_pad
        self.mask_state_packed = _mm(
            "mask_state_packed", (B, num_steps, mask_packed_len), np.uint8
        )
        self.top_k_indices = _mm("top_k_indices", (B, num_steps, L_resp, K), np.int32)
        self.top_k_logprobs_bf16 = _mm(
            "top_k_logprobs_bf16", (B, num_steps, L_resp, K), np.uint16
        )
        self.neg_log_tail_mass_bf16 = _mm(
            "neg_log_tail_mass_bf16", (B, num_steps, L_resp), np.uint16
        )
        self.top1_conf_bf16 = _mm("top1_conf_bf16", (B, num_steps, L_resp), np.uint16)
        self.entropy_bf16 = _mm("entropy_bf16", (B, num_steps, L_resp), np.uint16)

    # ------------------------------------------------------------------
    # Per-sample / per-step write hooks
    # ------------------------------------------------------------------

    def set_sample(
        self,
        i: int,
        *,
        sample_id: int,
        prompt_len: int,
        final_sequence: np.ndarray,
    ) -> None:
        self.sample_ids[i] = sample_id
        self.prompt_lens[i] = prompt_len
        T = int(final_sequence.shape[0])
        self.final_sequences[i, :T] = final_sequence

    def set_step(
        self,
        i: int,
        step_idx: int,
        *,
        mask_state: np.ndarray,
        top_k_indices: np.ndarray,
        top_k_logprobs: np.ndarray,
        neg_log_tail_mass: np.ndarray,
        top1_conf: np.ndarray,
        entropy: np.ndarray,
    ) -> None:
        # mask_state: bool [L_resp]; pack here so we touch a small slice.
        packed = pack_mask_bits(mask_state.astype(bool, copy=False)[None])[0]
        self.mask_state_packed[i, step_idx] = packed
        self.top_k_indices[i, step_idx] = top_k_indices.astype(np.int32, copy=False)
        self.top_k_logprobs_bf16[i, step_idx] = fp32_numpy_to_uint16_bf16(
            top_k_logprobs.astype(np.float32, copy=False)
        )
        self.neg_log_tail_mass_bf16[i, step_idx] = fp32_numpy_to_uint16_bf16(
            neg_log_tail_mass.astype(np.float32, copy=False)
        )
        self.top1_conf_bf16[i, step_idx] = fp32_numpy_to_uint16_bf16(
            top1_conf.astype(np.float32, copy=False)
        )
        self.entropy_bf16[i, step_idx] = fp32_numpy_to_uint16_bf16(
            entropy.astype(np.float32, copy=False)
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "StreamingShardWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        for name in self._ARRAY_NAMES:
            arr = getattr(self, name, None)
            if arr is not None:
                arr.flush()
                # Drop our reference so the OS can release the mmap. The
                # underlying memmap object is GC'd on next collection.
                setattr(self, name, None)


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
    """Build the noised input that the teacher saw at the start of a step."""
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
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _git_commit_hash() -> Optional[str]:
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
    """Tiny round-trip check; runnable without a GPU."""
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
        shard_dir = os.path.join(td, "shard_0000")
        write_shard(shard_dir, shard)
        round_tripped = read_shard(shard_dir, mmap=True, materialize_fp32=True)

        # Also exercise the mmap-open path used by training.
        mm = open_shard_arrays(shard_dir)
        # Slicing a memmap should work without loading the whole array.
        slice_uint16 = mm["top_k_logprobs_bf16"][0, 0]  # [L_resp, K] uint16
        assert slice_uint16.shape == (L_resp, K)

    assert np.array_equal(shard.sample_ids, round_tripped.sample_ids)
    assert np.array_equal(shard.prompt_lens, round_tripped.prompt_lens)
    assert np.array_equal(shard.final_sequences, round_tripped.final_sequences)
    assert np.array_equal(shard.mask_state, round_tripped.mask_state)
    assert np.array_equal(shard.top_k_indices, round_tripped.top_k_indices)
    for name in ("top_k_logprobs", "neg_log_tail_mass", "top1_conf", "entropy"):
        a = getattr(shard, name)
        b = getattr(round_tripped, name)
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-3, err_msg=name)

    print("cache_io._sanity OK: shard round-trip preserves all fields within bf16 precision")


if __name__ == "__main__":
    _sanity()
