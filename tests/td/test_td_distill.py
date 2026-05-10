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
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Fake cache builder -- uses real softmax so top-K + tail are a valid
# decomposition of one teacher distribution per position.
# ---------------------------------------------------------------------------


def _build_fake_cache(
    cache_dir: str,
    num_samples: int = 4,
    teacher_steps: int = 8,
    max_response_len: int = 6,
    top_k: int = 4,
    shard_size: int = 2,
    vocab_size: int = 50,
    seed: int = 0,
) -> CacheManifest:
    """Write a tiny manifest + shards. Teacher distributions come from a real
    random softmax so ``top_k_logprobs`` and ``neg_log_tail_mass`` are
    *the* top-K and 1 - sum(top-K) of one underlying distribution each.
    """
    rng = np.random.default_rng(seed)
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
        mask_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 2,
        bos_token_id=vocab_size - 3,
        vocab_size=vocab_size,
        tokenizer_name_or_path="/fake/teacher",
    )

    shards: list[ShardInfo] = []

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
        # More masked at early steps, fewer later (rough analogue of teacher).
        mask_state = np.zeros((B, teacher_steps, max_response_len), dtype=bool)
        for b in range(B):
            for t in range(teacher_steps):
                n_masked = max(0, max_response_len - t)
                positions = rng.permutation(max_response_len)[:n_masked]
                mask_state[b, t, positions] = True

        # Teacher distribution per (sample, step, position): random softmax
        # with adjustable peakedness so top-K usually covers most mass.
        logits = (
            rng.standard_normal((B, teacher_steps, max_response_len, vocab_size))
            * 2.5
        ).astype(np.float32)
        probs_t = torch.from_numpy(logits).softmax(dim=-1)  # [B, T, L, V]
        top_k_probs_t, top_k_idx_t = probs_t.topk(top_k, dim=-1)
        tail_mass_t = (1.0 - top_k_probs_t.sum(-1)).clamp(min=1e-30)
        top_k_logprobs_t = top_k_probs_t.clamp(min=1e-30).log()
        neg_log_tail_t = -tail_mass_t.log()
        top1_conf_t = top_k_probs_t[..., 0]
        entropy_t = -(probs_t * probs_t.clamp(min=1e-30).log()).sum(-1)

        shard_data = ShardData(
            sample_ids=sample_ids,
            prompt_lens=prompt_lens,
            final_sequences=final_seq,
            mask_state=mask_state,
            top_k_indices=top_k_idx_t.numpy().astype(np.int32),
            top_k_logprobs=top_k_logprobs_t.numpy().astype(np.float32),
            neg_log_tail_mass=neg_log_tail_t.numpy().astype(np.float32),
            top1_conf=top1_conf_t.numpy().astype(np.float32),
            entropy=entropy_t.numpy().astype(np.float32),
        )
        rel = f"shard_{shard_idx:04d}"  # directory, not a file
        write_shard(os.path.join(cache_dir, rel), shard_data)
        shards.append(
            ShardInfo(
                path=rel,
                sample_ids=[int(x) for x in sample_ids],
                t_shard=T_shard,
            )
        )

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


def _quiet_dataset(cache_dir, **kwargs) -> TrajectoryCacheDataset:
    """Wrap dataset construction in a warnings filter -- the fake cache
    has a fake git commit which would otherwise trigger a UserWarning
    during every test that loads one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TrajectoryCacheDataset(cache_dir=cache_dir, **kwargs)


# ---------------------------------------------------------------------------
# Dataset shape / index math
# ---------------------------------------------------------------------------


def test_dataset_len_and_item_shapes():
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=4, teacher_steps=8, max_response_len=6, top_k=4)
        ds = _quiet_dataset(td, student_steps=4, teacher_stride=2)
        assert len(ds) == 4 * 4
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
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=2, teacher_steps=8, max_response_len=4, top_k=4)
        ds = _quiet_dataset(td, student_steps=4, teacher_stride=2)
        for i in range(len(ds)):
            item = ds[i]
            assert item["student_step"] * 2 == item["teacher_step"]


def test_dataset_stride_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=2, teacher_steps=8)
        with pytest.raises(ValueError, match="teacher_stride"):
            _quiet_dataset(td, student_steps=3, teacher_stride=3)


def test_dataset_sample_id_filter():
    with tempfile.TemporaryDirectory() as td:
        _build_fake_cache(td, num_samples=4, teacher_steps=8)
        kept = {0, 2}
        ds = _quiet_dataset(
            td, student_steps=4, teacher_stride=2, sample_id_filter=kept
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
        ds = _quiet_dataset(td, student_steps=4, teacher_stride=2)
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

    for b in range(B):
        valid_end = batch["prompt_lens"][b].item() + 6
        assert batch["attention_mask"][b, :valid_end].all()
        assert (batch["attention_mask"][b, valid_end:] == 0).all()


# ---------------------------------------------------------------------------
# Trainer compute_loss with stub models
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


def _make_trainer_skeleton() -> TDDistillationTrainer:
    """Construct a trainer object without going through HF Trainer.__init__.

    HF Trainer's full constructor wants an output_dir and a real
    accelerate set-up. We only call ``compute_loss`` directly, which
    doesn't touch any of that, so a bare ``__new__`` is sufficient.
    """
    return TDDistillationTrainer.__new__(TDDistillationTrainer)


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
        ds = _quiet_dataset(td, student_steps=4, teacher_stride=2)
        items = [ds[i] for i in (0, 1, 2, 3)]

    collator = TDDistillCollator(pad_token_id=0)
    batch = collator(items)

    model = _StubLM(vocab_size=vocab_size).eval()
    for p in model.parameters():
        p.requires_grad = True

    trainer = _make_trainer_skeleton()
    loss = trainer.compute_loss(model, batch)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    loss.backward()
    grad_norm = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0


def test_compute_loss_small_when_student_matches_teacher():
    """If the student emits exactly the teacher's softmax distribution at
    each supervised position, coarse KL is bounded by a small constant
    (only the 'student tail flatness vs teacher tail shape' term survives).
    """
    vocab_size = 40
    with tempfile.TemporaryDirectory() as td:
        # Build a cache with a known underlying logits seed so we can
        # reconstruct teacher's full distribution exactly per position.
        rng_seed = 7
        _build_fake_cache(
            td,
            num_samples=1,
            teacher_steps=4,
            max_response_len=4,
            top_k=4,
            shard_size=1,
            vocab_size=vocab_size,
            seed=rng_seed,
        )
        ds = _quiet_dataset(td, student_steps=2, teacher_stride=2)
        items = [ds[0]]

    collator = TDDistillCollator(pad_token_id=0)
    batch = collator(items)

    sup = batch["supervised_mask"][0]  # [L_resp]
    if sup.sum() == 0:
        pytest.skip("no supervised positions in fake batch (rare draw)")

    # Reconstruct the teacher's underlying full distribution from the
    # cache's top-K + tail. We KNOW the cache was built from a softmax,
    # so for each position we have:
    #   teacher_p[top_K_idx[k]] = exp(top_k_logprobs[k])
    #   sum over remaining positions = exp(-neg_log_tail)
    # We'll spread the tail UNIFORMLY across the non-top-K vocabulary;
    # this is an approximation of the true tail (which had more
    # structure), so the resulting student distribution differs slightly
    # from the teacher's. The test tolerance reflects this.
    teacher_idx = batch["teacher_top_k_indices"][0]
    teacher_lp = batch["teacher_top_k_logprobs"][0]
    teacher_neg_log_tail = batch["teacher_neg_log_tail"][0]
    L_resp = int(sup.shape[0])
    K = teacher_idx.shape[-1]
    V = vocab_size

    target_logits = torch.full((L_resp, V), -50.0)
    for p_idx in range(L_resp):
        for k in range(K):
            target_logits[p_idx, teacher_idx[p_idx, k]] = teacher_lp[p_idx, k]
        log_tail = -teacher_neg_log_tail[p_idx]
        n_tail = V - K
        if n_tail > 0:
            log_per_tail_tok = log_tail - torch.log(torch.tensor(float(n_tail)))
            tail_mask = torch.ones(V, dtype=torch.bool)
            tail_mask[teacher_idx[p_idx]] = False
            target_logits[p_idx, tail_mask] = log_per_tail_tok

    class _PerfectLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None):
            B, T = input_ids.shape
            logits = torch.zeros(B, T, V)
            pl = batch["prompt_lens"][0].item()
            logits[0, pl : pl + L_resp] = target_logits + self.dummy * 0.0
            return SimpleNamespace(logits=logits)

    model = _PerfectLM()
    trainer = _make_trainer_skeleton()
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    # The student's distribution is teacher's top-K (exact) + uniform tail
    # (approximation). The KL term over the top-K bucket is essentially 0;
    # the tail term reflects only the entropy gap between teacher's true
    # tail and our uniform reconstruction. < 1.0 is a comfortable bound.
    assert loss.item() < 1.0
