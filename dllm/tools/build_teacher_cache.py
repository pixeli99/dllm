"""Build a teacher trajectory cache for ReEntry-TD distillation.

How to run on the cluster (Phase 1, full 1k cache)::

    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    cd /lustre/projects/polyullm/lipengxiang_tmp/dllm
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 \
        --cpus-per-task=24 --time=24:00:00 \
        python /lustre/projects/polyullm/lipengxiang_tmp/dllm/dllm/tools/build_teacher_cache.py \
            --teacher_ckpt /lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-baseline/checkpoint-final \
            --dataset_path /lustre/projects/polyullm/lipengxiang_tmp/dllm/.data/sft/llada/openmath2-500k \
            --dataset_split train \
            --num_samples 1000 \
            --teacher_steps 256 \
            --max_response_len 256 \
            --top_k 64 \
            --shard_size 100 \
            --micro_batch_size 4 \
            --output_dir /lustre/projects/polyullm/lipengxiang_tmp/dllm/.cache/teacher_traj/openmath2-1k-256step \
            --max_prompt_len 768

How to do the mandatory mini-cache pre-flight (REENTRY_TD_SPEC.md §5.4)::

    same as above but --num_samples 50 and --output_dir suffix mini

What this does:

1. Loads the teacher SFT checkpoint and tokenizer.
2. Loads a preprocessed SFT DatasetDict on disk and extracts prompts
   (the prefix where ``labels == -100``).
3. Selects ``num_samples`` prompts that fit ``max_prompt_len``.
4. Groups them into shards of ``shard_size``; for each shard,
   micro-batches through the deterministic sampler.
5. Writes one ``.npz`` per shard plus a ``manifest.json`` that pins the
   git commit, sampler config hash, torch/cuda versions, and the
   per-shard sample IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

# Local imports
import dllm  # registers custom configs/models  (per AGENTS.md: import dllm matters)
from dllm.core.samplers.mdlm_deterministic import (
    DeterministicMDLMSampler,
    DeterministicMDLMSamplerConfig,
    sampler_config_hash_payload,
)
from dllm.pipelines.llada_looped.cache_io import (
    CacheManifest,
    ShardData,
    ShardInfo,
    TrajectoryCacheConfig,
    compute_sampler_config_hash,
    utcnow_iso,
    write_shard,
    _git_commit_hash,
)
from dllm.utils import get_model, get_tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # --- Inputs ---
    p.add_argument("--teacher_ckpt", type=str, required=True,
                   help="Path or HF id of the frozen task-SFT teacher.")
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to a preprocessed SFT DatasetDict (load_from_disk).")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--max_prompt_len", type=int, default=768,
                   help="Skip samples whose prompt is longer than this.")
    p.add_argument("--min_prompt_len", type=int, default=4,
                   help="Skip samples whose prompt is shorter than this (degenerate).")

    # --- Cache shape ---
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--teacher_steps", type=int, default=256)
    p.add_argument("--max_response_len", type=int, default=256)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--block_size", type=int, default=0,
                   help="0 means use max_response_len (single block).")

    # --- Sharding / runtime ---
    p.add_argument("--shard_size", type=int, default=100)
    p.add_argument("--micro_batch_size", type=int, default=4,
                   help="How many prompts to forward at once. "
                        "Determinism within a fixed micro_batch_size is bit-exact, but "
                        "cross-batch-size determinism is not guaranteed (FlashAttention etc).")
    p.add_argument("--seed_base", type=int, default=20260510)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for the teacher forward.")

    # --- Output ---
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--allow_overwrite", action="store_true")
    p.add_argument("--no_chat_template", action="store_true",
                   help="Don't validate that prompts look chat-template-formed. "
                        "Useful for non-instruct datasets.")

    # --- Audit (mini-cache pre-flight) ---
    p.add_argument("--audit", action="store_true",
                   help="After writing, run the top-K mass audit "
                        "(>=75%% of supervised positions should have top-K mass >= 0.90). "
                        "Recommended on the 50-sample mini-cache.")
    p.add_argument("--audit_min_pass_frac", type=float, default=0.75)
    p.add_argument("--audit_min_top_k_mass", type=float, default=0.90)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def _extract_prompt(row: dict) -> tuple[np.ndarray | None, int | None]:
    """Return (prompt_ids, prompt_end) or (None, None) on rejection."""
    input_ids = np.asarray(row["input_ids"], dtype=np.int64)
    labels = np.asarray(row["labels"], dtype=np.int64)
    if input_ids.shape != labels.shape:
        return None, None
    nonprompt = np.where(labels != -100)[0]
    if nonprompt.size == 0:
        return None, None  # all prompt, no response to learn from
    prompt_end = int(nonprompt[0])
    if prompt_end == 0:
        return None, None  # no prompt
    return input_ids[:prompt_end], prompt_end


def select_samples(
    ds_split,
    num_samples: int,
    max_prompt_len: int,
    min_prompt_len: int,
) -> list[dict]:
    """Walk the dataset in order, keep prompts that pass length filters."""
    samples: list[dict] = []
    n_seen = 0
    for i in range(len(ds_split)):
        n_seen += 1
        row = ds_split[i]
        prompt_ids, _ = _extract_prompt(row)
        if prompt_ids is None:
            continue
        L = prompt_ids.shape[0]
        if L < min_prompt_len or L > max_prompt_len:
            continue
        samples.append({"sample_id": i, "prompt_ids": prompt_ids})
        if len(samples) == num_samples:
            break

    if len(samples) < num_samples:
        raise RuntimeError(
            f"only collected {len(samples)} usable prompts after scanning {n_seen} rows; "
            f"requested {num_samples}. Try raising --max_prompt_len, lowering "
            "--min_prompt_len, or pointing at a larger split."
        )
    return samples


# ---------------------------------------------------------------------------
# Shard assembly
# ---------------------------------------------------------------------------


def build_shard(
    samples: list[dict],
    sampler: DeterministicMDLMSampler,
    sampler_cfg: DeterministicMDLMSamplerConfig,
    micro_batch_size: int,
    eos_id: int,
) -> tuple[ShardData, int]:
    """Run the sampler on one shard's worth of samples, return ShardData.

    Returns ``(shard_data, t_shard)`` where ``t_shard = max prompt_len + L_resp``.
    """
    B = len(samples)
    L_resp = sampler_cfg.max_response_len
    K = sampler_cfg.top_k
    num_steps = sampler_cfg.teacher_steps

    max_prompt_len = max(int(s["prompt_ids"].shape[0]) for s in samples)
    t_shard = max_prompt_len + L_resp

    sample_ids = np.zeros(B, dtype=np.int64)
    prompt_lens = np.zeros(B, dtype=np.int32)
    final_sequences = np.full((B, t_shard), eos_id, dtype=np.int64)
    mask_state = np.zeros((B, num_steps, L_resp), dtype=bool)
    top_k_indices = np.zeros((B, num_steps, L_resp, K), dtype=np.int32)
    top_k_logprobs = np.zeros((B, num_steps, L_resp, K), dtype=np.float32)
    neg_log_tail_mass = np.zeros((B, num_steps, L_resp), dtype=np.float32)
    top1_conf = np.zeros((B, num_steps, L_resp), dtype=np.float32)
    entropy = np.zeros((B, num_steps, L_resp), dtype=np.float32)

    for batch_start in range(0, B, micro_batch_size):
        batch_end = min(batch_start + micro_batch_size, B)
        batch = samples[batch_start:batch_end]
        prompts = [torch.as_tensor(s["prompt_ids"], dtype=torch.long) for s in batch]

        out = sampler.sample_with_trajectory(prompts, sampler_cfg)

        seq_np = out.sequences.cpu().numpy()  # [B_micro, T_micro]
        for j, s in enumerate(batch):
            i = batch_start + j
            sample_ids[i] = s["sample_id"]
            prompt_lens[i] = out.prompt_lens[j]
            T_micro = seq_np.shape[1]
            final_sequences[i, :T_micro] = seq_np[j]
            # Trailing positions stay as EOS.

            traj = out.trajectories[j]
            if len(traj) != num_steps:
                raise RuntimeError(
                    f"sample {s['sample_id']}: got {len(traj)} trajectory steps "
                    f"but expected {num_steps}. Did the schedule emit zero-step "
                    "rows? Check teacher_steps vs max_response_len."
                )
            for step_idx, ts in enumerate(traj):
                mask_state[i, step_idx] = ts.mask_state
                top_k_indices[i, step_idx] = ts.top_k_indices
                top_k_logprobs[i, step_idx] = ts.top_k_logprobs
                neg_log_tail_mass[i, step_idx] = ts.neg_log_tail_mass
                top1_conf[i, step_idx] = ts.top1_conf
                entropy[i, step_idx] = ts.entropy

    return (
        ShardData(
            sample_ids=sample_ids,
            prompt_lens=prompt_lens,
            final_sequences=final_sequences,
            mask_state=mask_state,
            top_k_indices=top_k_indices,
            top_k_logprobs=top_k_logprobs,
            neg_log_tail_mass=neg_log_tail_mass,
            top1_conf=top1_conf,
            entropy=entropy,
        ),
        t_shard,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.allow_overwrite:
            print(
                f"[error] output_dir {output_dir} is non-empty. "
                "Pass --allow_overwrite to clobber.",
                file=sys.stderr,
            )
            return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve dtype ----
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ---- Tokenizer ----
    print(f"[info] loading tokenizer from {args.teacher_ckpt}", flush=True)
    tokenizer = get_tokenizer(model_name_or_path=args.teacher_ckpt)
    if tokenizer.mask_token_id is None:
        raise RuntimeError(
            "tokenizer has no mask_token_id; LLaDA-style models require it. "
            "Make sure --teacher_ckpt points at a LLaDA SFT checkpoint."
        )

    # ---- Teacher model ----
    print(f"[info] loading teacher model ({args.dtype})", flush=True)
    model = get_model(model_name_or_path=args.teacher_ckpt, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if args.device != "cpu" and not next(model.parameters()).is_cuda:
        model = model.to(args.device)

    # ---- Dataset ----
    print(f"[info] loading dataset from {args.dataset_path} (split={args.dataset_split})", flush=True)
    ds = load_from_disk(args.dataset_path)
    if args.dataset_split not in ds:
        raise RuntimeError(
            f"split '{args.dataset_split}' missing in dataset; available: {list(ds.keys())}"
        )
    ds_split = ds[args.dataset_split]

    print(f"[info] selecting {args.num_samples} prompts (max_prompt_len={args.max_prompt_len})", flush=True)
    samples = select_samples(
        ds_split=ds_split,
        num_samples=args.num_samples,
        max_prompt_len=args.max_prompt_len,
        min_prompt_len=args.min_prompt_len,
    )

    # ---- Sampler ----
    block_size = args.block_size if args.block_size > 0 else None
    sampler_cfg = DeterministicMDLMSamplerConfig(
        teacher_steps=args.teacher_steps,
        max_response_len=args.max_response_len,
        top_k=args.top_k,
        block_size=block_size,
        record_trajectory=True,
    )
    sampler = DeterministicMDLMSampler(model=model, tokenizer=tokenizer)

    # ---- Cache config + manifest skeleton ----
    cache_cfg = TrajectoryCacheConfig(
        teacher_model_path=args.teacher_ckpt,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
        teacher_steps=args.teacher_steps,
        max_response_len=args.max_response_len,
        top_k=args.top_k,
        block_size=block_size if block_size is not None else args.max_response_len,
        shard_size=args.shard_size,
        seed_base=args.seed_base,
        mask_token_id=int(tokenizer.mask_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        bos_token_id=int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else -1,
        vocab_size=int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        tokenizer_name_or_path=args.teacher_ckpt,
    )
    sampler_hash = compute_sampler_config_hash(sampler_config_hash_payload(sampler_cfg))
    git_hash = _git_commit_hash() or "unknown"
    cuda_version = torch.version.cuda  # type: ignore[attr-defined]

    manifest = CacheManifest(
        config=cache_cfg,
        git_commit_hash=git_hash,
        sampler_config_hash=sampler_hash,
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        created_at=utcnow_iso(),
        shards=[],
    )

    # ---- Build shards ----
    num_shards = (args.num_samples + args.shard_size - 1) // args.shard_size
    eos_id = int(tokenizer.eos_token_id)
    print(
        f"[info] generating cache: {args.num_samples} samples, "
        f"{args.teacher_steps} steps, top_k={args.top_k}, "
        f"shard_size={args.shard_size} -> {num_shards} shards",
        flush=True,
    )

    t_global = time.time()
    for shard_idx in range(num_shards):
        start = shard_idx * args.shard_size
        end = min(start + args.shard_size, args.num_samples)
        shard_samples = samples[start:end]
        rel_path = f"shard_{shard_idx:04d}.npz"
        shard_path = output_dir / rel_path

        print(
            f"[info] shard {shard_idx + 1}/{num_shards}: "
            f"{len(shard_samples)} samples (sample_ids "
            f"{shard_samples[0]['sample_id']}..{shard_samples[-1]['sample_id']})",
            flush=True,
        )
        t0 = time.time()
        shard_data, t_shard = build_shard(
            samples=shard_samples,
            sampler=sampler,
            sampler_cfg=sampler_cfg,
            micro_batch_size=args.micro_batch_size,
            eos_id=eos_id,
        )
        t_sample = time.time() - t0
        print(f"[info]   sampled in {t_sample:.1f}s", flush=True)

        t1 = time.time()
        md5 = write_shard(shard_path, shard_data)
        t_write = time.time() - t1
        size_mb = shard_path.stat().st_size / (1 << 20)
        print(f"[info]   wrote {size_mb:.1f} MB in {t_write:.1f}s, md5={md5[:8]}", flush=True)

        manifest.shards.append(
            ShardInfo(
                path=rel_path,
                sample_ids=[int(s["sample_id"]) for s in shard_samples],
                t_shard=t_shard,
                md5=md5,
            )
        )
        manifest.save(output_dir)  # incremental save in case of crash

    elapsed = time.time() - t_global
    print(f"[info] done. {num_shards} shards in {elapsed:.1f}s", flush=True)

    # ---- Audit pass (Phase 1 mini-cache requirement) ----
    if args.audit:
        ok = run_audit(
            output_dir=output_dir,
            manifest=manifest,
            min_pass_frac=args.audit_min_pass_frac,
            min_top_k_mass=args.audit_min_top_k_mass,
        )
        if not ok:
            print(
                "[error] audit FAILED. Consider re-running with a larger --top_k.",
                file=sys.stderr,
            )
            return 3

    return 0


def run_audit(
    output_dir: Path,
    manifest: CacheManifest,
    min_pass_frac: float,
    min_top_k_mass: float,
) -> bool:
    """Audit: in >= min_pass_frac of supervised positions, top-K mass >= min_top_k_mass.

    "Supervised" here means: response positions that were still masked
    at the start of that step (these are what student loss touches).
    """
    from dllm.pipelines.llada_looped.cache_io import (
        read_shard,
        uint16_bf16_to_fp32_numpy,
    )

    print(
        f"[audit] checking top-K mass >= {min_top_k_mass} on supervised positions, "
        f"target frac >= {min_pass_frac}",
        flush=True,
    )

    pass_count = 0
    total_count = 0
    for shard_idx, shard_info in enumerate(manifest.shards):
        path = output_dir / shard_info.path
        shard = read_shard(path, mmap=True, materialize_fp32=True)
        # Compute top-K mass = exp(top_k_logprobs).sum(axis=-1)
        top_k_mass = np.exp(shard.top_k_logprobs).sum(axis=-1)  # [B, num_steps, L_resp]
        # Supervised positions = currently masked positions only
        supervised = shard.mask_state  # [B, num_steps, L_resp] bool
        passes = (top_k_mass >= min_top_k_mass) & supervised
        pass_count += int(passes.sum())
        total_count += int(supervised.sum())
        print(
            f"[audit] shard {shard_idx}: {int(passes.sum())} / {int(supervised.sum())} "
            f"pass ({passes.sum() / max(supervised.sum(), 1):.2%})",
            flush=True,
        )

    if total_count == 0:
        print("[audit] no supervised positions found?!", file=sys.stderr)
        return False

    overall_frac = pass_count / total_count
    print(
        f"[audit] overall: {pass_count} / {total_count} pass ({overall_frac:.2%}). "
        f"target {min_pass_frac:.0%}.",
        flush=True,
    )
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "min_pass_frac": min_pass_frac,
                "min_top_k_mass": min_top_k_mass,
                "pass_count": pass_count,
                "total_count": total_count,
                "overall_frac": overall_frac,
                "passed": overall_frac >= min_pass_frac,
            },
            indent=2,
        )
    )
    return overall_frac >= min_pass_frac


if __name__ == "__main__":
    sys.exit(main())
