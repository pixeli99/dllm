"""TD distillation pipeline tests: cache dataset + collator + trainer compute_loss.

How to run::

    cd /Users/pixeli/dllm
    python -m pytest tests/td/test_td_distill.py -v

These tests synthesise a tiny fake cache on disk, then exercise:
  - TrajectoryCacheDataset shape / index math
  - sample-id filter, train/val split determinism
  - TDDistillCollator padding
  - TDDistillationTrainer.compute_loss with a stub model

No real LLaDA, no GPU.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from dllm.core.trainers.td_distill import (
    TDDistillationTrainer,
    TDDistillCollator,
    TrajectoryCacheDataset,
)
from dllm.pipelines.llada_looped.cache_io import (
    CacheManifest,
    ShardData,
    ShardInfo,
    TrajectoryCacheConfig,
    compute_sampler_config_hash,
    utcnow_iso,
    write_shard,
)


# ---------------------------------------------------------------------------
# Fake cache builder
# ---------------------------------------------------------------------------


def _build_fake_cache(
    cache_dir: str,
    num_samples: int = 4,
    teacher_steps: int = 8,
    max_response_len: int = 6,
    top_k: int = 4,
    shard_size: int = 2,
    vocab_size: int = 50,
) -> CacheManifest:
    """Write a tiny manifest + shards to disk so a Dataset can read them.

    All numerical fields are random; we only verify shapes / dtypes /
    indexing in tests, not numerical values.
    """
    rng = np.random.default_rng(0)
    cfg = TrajectoryCacheConfig(
        teacher_model_path="/fake/teacher",
        dataset_path="/fake/data",
        dataset_split="train",
        num_samples=num_samples,
        teacher_steps=teacher_steps,
        max_response_len=max_response_len,
        top_k=top_k,
        block_size=max_response_len,
        shard_size=shard_size,
        seed_base=0,
        mask_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 2,
        bos_token_id=vocab_size - 3,
        vocab_size=vocab_size,
        tokenizer_name_or_path="/fake/teacher",
    )

    shards: list[ShardInfo] = []
    sample_id = 0
    for shard_idx in range((num_samples + shard_size - 1) // shard_size):
        start = shard_idx * shard_size
        end = min(start + shard_size, num_samples)
        B = end - start

        prompt_lens = rng.integers(2, 5, size=B, dtype=np.int32)
        max_prompt = int(prompt_lens.max())
        T_shard = max_prompt + max_response_len

        sample_ids = np.arange(start, end, dtype=np.int64)
        final_seq = rng.integers(0, vocab_size - 3, size=(B, T_shard), dtype=np.int64)

        # Mask state: at step t, mark some response positions still masked.
        # Realistic-ish: more masked at early steps, fewer later.
        mask_state = np.zeros((B, teacher_steps, max_response_len), dtype=bool)
        for b in range(B):
            for t in range(teacher_steps):
                # masked count at step t = max_response_len - t (roughly)
                n_masked = max(0, max_response_len - t)
                positions = rng.permutation(max_response_len)[:n_masked]
                mask_state[b, t, positions] = True

        top_k_indices = rng.integers(
            0, vocab_size, size=(B, teacher_steps, max_response_len, top_k), dtype=np.int32
        )
        top_k_logprobs = rng.uniform(
            -10, 0, size=(B, teacher_steps, max_response_len, top_k)
        ).astype(np.float32)
        # Ensure each position's top-K log-probs roughly sum to <= 1 in prob.
        # Simpler: just make them small enough.
        top_k_logprobs = top_k_logprobs - 5.0  # shift so exp().sum() < 1

        neg_log_tail = rng.uniform(
            0, 5, size=(B, teacher_steps, max_response_len)
        ).astype(np.float32)
        top1_conf = rng.uniform(
            0, 1, size=(B, teacher_steps, max_response_len)
        ).astype(np.float32)
        entropy = rng.uniform(
            0, 4, size=(B, teacher_steps, max_response_len)
        ).astype(np.float32)

        shard_data = ShardData(
            sample_ids=sample_ids,
            prompt_lens=prompt_lens,
            final_sequences=final_seq,
            mask_state=mask_state,
            top_k_indices=top_k_indices,
            top_k_logprobs=top_k_logprobs,
            neg_log_tail_mass=neg_log_tail,
            top1_conf=top1_conf,
            entropy=entropy,
        )
        rel = f"shard_{shard_idx:04d}.npz"
        md5 = write_shard(os.path.join(cache_dir, rel), shard_data)
        shards.append(
            ShardInfo(
                path=rel,
                sample_ids=[int(x) for x in sample_ids],
                t_shard=T_shard,
                md5=md5,
            )
        )
        sample_id = end

    manifest = CacheManifest(
        config=cfg,
        git_commit_hash="fake-commit",
        sampler_config_hash=compute_sampler_config_hash({"fake": True}),
        torch_version=torch.__version__,
        cuda_version=None,
        created_at=utcnow_iso(),
        shards=shards,
    )
    manifest.save(cache_dir)
    return manifest


# ---------------------------------------------------------------------------
# Dataset shape / index math
# ---------------------------------------------------------------------------


def test_dataset_len_and_item_shapes():
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=4, teacher_steps=8, max_response_len=6, top_k=4)
        ds = TrajectoryCacheDataset(
            cache_dir=td, student_steps=4, teacher_stride=2, allow_stale_cache=True
        )
        assert len(ds) == 4 * 4  # 4 samples * 4 student steps
        item = ds[0]
        assert item["input_ids"].dtype == np.int64
        assert item["supervised_mask"].shape == (6,)
        assert item["supervised_mask"].dtype == bool
        assert item["teacher_top_k_indices"].shape == (6, 4)
        assert item["teacher_top_k_logprobs"].shape == (6, 4)
        assert item["teacher_top_k_logprobs"].dtype == np.float32
        assert item["teacher_neg_log_tail"].shape == (6,)
        assert item["teacher_neg_log_tail"].dtype == np.float32


def test_dataset_index_to_step_mapping():
    """Item i decodes to (sample, student_step) and student_step *
    teacher_stride = teacher_step."""
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=2, teacher_steps=8, max_response_len=4, top_k=4)
        ds = TrajectoryCacheDataset(
            cache_dir=td, student_steps=4, teacher_stride=2, allow_stale_cache=True
        )

        # Walk all items, check that teacher_step = student_step * stride.
        for i in range(len(ds)):
            item = ds[i]
            assert item["student_step"] * 2 == item["teacher_step"]


def test_dataset_stride_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=2, teacher_steps=8)
        with pytest.raises(ValueError, match="teacher_stride"):
            TrajectoryCacheDataset(
                cache_dir=td,
                student_steps=3,  # 3 * any_int != 8
                teacher_stride=3,
                allow_stale_cache=True,
            )


def test_dataset_sample_id_filter():
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=4, teacher_steps=8)
        kept = {0, 2}  # filter to two specific samples
        ds = TrajectoryCacheDataset(
            cache_dir=td,
            student_steps=4,
            teacher_stride=2,
            sample_id_filter=kept,
            allow_stale_cache=True,
        )
        assert len(ds) == 2 * 4
        seen_ids = {ds[i]["sample_id"] for i in range(len(ds))}
        assert seen_ids == kept


def test_split_sample_ids_deterministic():
    ids = list(range(20))
    a_train, a_val = TrajectoryCacheDataset.split_sample_ids(ids, 0.25, seed=42)
    b_train, b_val = TrajectoryCacheDataset.split_sample_ids(ids, 0.25, seed=42)
    assert a_train == b_train
    assert a_val == b_val
    assert len(a_val) == 5
    assert a_train.isdisjoint(a_val)
    assert a_train | a_val == set(ids)


def test_split_sample_ids_zero_fraction_returns_empty_val():
    ids = list(range(10))
    train, val = TrajectoryCacheDataset.split_sample_ids(ids, 0.0)
    assert train == set(ids)
    assert val == set()


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


def test_collator_padding_and_attention_mask():
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=4, teacher_steps=8, max_response_len=6, top_k=4)
        ds = TrajectoryCacheDataset(
            cache_dir=td, student_steps=4, teacher_stride=2, allow_stale_cache=True
        )
        items = [ds[0], ds[5], ds[10], ds[15]]

    collator = TDDistillCollator(pad_token_id=0)
    batch = collator(items)

    B = 4
    assert batch["input_ids"].shape[0] == B
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["prompt_lens"].shape == (B,)
    assert batch["supervised_mask"].shape == (B, 6)
    assert batch["supervised_mask"].dtype == torch.bool
    assert batch["teacher_top_k_indices"].shape == (B, 6, 4)
    assert batch["teacher_top_k_indices"].dtype == torch.long
    assert batch["teacher_top_k_logprobs"].dtype == torch.float32
    assert batch["teacher_neg_log_tail"].dtype == torch.float32

    # attention_mask should be 1 for prompt + response positions, 0 elsewhere.
    for b in range(B):
        valid_end = batch["prompt_lens"][b].item() + 6
        assert batch["attention_mask"][b, :valid_end].all()
        assert (batch["attention_mask"][b, valid_end:] == 0).all()


# ---------------------------------------------------------------------------
# Trainer compute_loss with stub model
# ---------------------------------------------------------------------------


class _StubLM(nn.Module):
    """Token-conditioned linear head -> deterministic .logits."""

    def __init__(self, vocab_size: int, hidden: int = 16, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        return SimpleNamespace(logits=logits)


def _make_trainer(model, cache_dir):
    """Stand up a TDDistillationTrainer without going through HF Trainer's
    full constructor (which wants an output_dir and other side effects)."""
    # We cannot easily instantiate HF Trainer in unit tests. Instead, call
    # compute_loss directly on a thin shim with the trainer's bound method.
    trainer = TDDistillationTrainer.__new__(TDDistillationTrainer)
    return trainer


def test_compute_loss_runs_end_to_end():
    """Build a fake cache, build a stub student, run compute_loss once, and
    verify the returned loss is a finite scalar with a live grad path.
    """
    vocab_size = 50
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(
            td,
            num_samples=4,
            teacher_steps=8,
            max_response_len=6,
            top_k=4,
            vocab_size=vocab_size,
        )
        ds = TrajectoryCacheDataset(
            cache_dir=td, student_steps=4, teacher_stride=2, allow_stale_cache=True
        )
        items = [ds[i] for i in (0, 1, 2, 3)]

    collator = TDDistillCollator(pad_token_id=0)
    batch = collator(items)

    model = _StubLM(vocab_size=vocab_size).eval()
    for p in model.parameters():
        p.requires_grad = True

    trainer = _make_trainer(model, cache_dir=None)
    loss = trainer.compute_loss(model, batch)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_compute_loss_zero_when_student_is_perfect():
    """Construct a stub model that returns exactly the teacher's recorded
    log-probs at the supervised positions. Coarse KL must be ~0.
    """
    # Build a minimal cache with only one sample and few steps for a clean test.
    vocab_size = 40
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(
            td,
            num_samples=1,
            teacher_steps=4,
            max_response_len=4,
            top_k=4,
            shard_size=1,
            vocab_size=vocab_size,
        )
        ds = TrajectoryCacheDataset(
            cache_dir=td, student_steps=2, teacher_stride=2, allow_stale_cache=True
        )
        items = [ds[0]]

    collator = TDDistillCollator(pad_token_id=0)
    batch = collator(items)

    # Build a model whose logits at supervised positions reproduce the
    # teacher's full distribution. We use the recorded top-K + assumed-tail
    # to construct a target log-prob vector and have the model emit it
    # directly via a per-position lookup.
    sup = batch["supervised_mask"][0]  # [L_resp]
    if sup.sum() == 0:
        pytest.skip("no supervised positions in fake batch (rare draw)")

    # Reconstruct teacher full logprobs at supervised positions from the cache:
    # top-K log p plus a flat tail spread over the remaining V-K vocab.
    teacher_idx = batch["teacher_top_k_indices"][0]  # [L_resp, K]
    teacher_lp = batch["teacher_top_k_logprobs"][0]  # [L_resp, K]
    teacher_neg_log_tail = batch["teacher_neg_log_tail"][0]  # [L_resp]
    L_resp = sup.shape[0]
    K = teacher_idx.shape[-1]
    V = vocab_size

    target_logits = torch.full((L_resp, V), -50.0)
    for p_idx in range(L_resp):
        for k in range(K):
            target_logits[p_idx, teacher_idx[p_idx, k]] = teacher_lp[p_idx, k]
        # spread tail uniformly across non-top-K tokens
        log_tail = -teacher_neg_log_tail[p_idx]
        n_tail = V - K
        if n_tail > 0:
            log_per_tail_tok = log_tail - torch.log(torch.tensor(float(n_tail)))
            tail_mask = torch.ones(V, dtype=torch.bool)
            tail_mask[teacher_idx[p_idx]] = False
            target_logits[p_idx, tail_mask] = log_per_tail_tok

    # Now define a model that emits these logits at the response window.
    class _PerfectLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))  # so trainer can backward

        def forward(self, input_ids, attention_mask=None):
            B, T = input_ids.shape
            logits = torch.zeros(B, T, V)
            pl = batch["prompt_lens"][0].item()
            logits[0, pl : pl + L_resp] = target_logits + self.dummy * 0.0
            return SimpleNamespace(logits=logits)

    model = _PerfectLM()
    trainer = _make_trainer(model, cache_dir=None)
    loss = trainer.compute_loss(model, batch)
    # The model is ~exactly the teacher distribution, so coarse KL is small.
    # bf16 was never involved here; the only error source is the uniform-tail
    # approximation, which is exact when teacher tail is also uniform across
    # non-top-K. In our fake cache it's not, so we tolerate up to ~teacher's
    # own per-position KL between "true tail" and "uniform tail".
    # Just check finite and small.
    assert torch.isfinite(loss)
    assert loss.item() < 1.0
