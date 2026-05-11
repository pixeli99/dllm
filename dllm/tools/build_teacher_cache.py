"""Build a teacher trajectory cache for ReEntry-TD distillation.

Single GPU::

    python -m dllm.tools.build_teacher_cache \\
        --teacher_ckpt /path/to/teacher \\
        --dataset_path .data/sft/llada/openmath2-500k \\
        --num_samples 1000 \\
        --output_dir .cache/teacher_traj/openmath2-1k-256step

Multi-GPU (data-parallel, one process per GPU)::

    torchrun --standalone --nproc_per_node=8 \\
        -m dllm.tools.build_teacher_cache \\
        --teacher_ckpt /path/to/teacher \\
        --dataset_path .data/sft/llada/openmath2-500k \\
        --num_samples 1000 \\
        --output_dir .cache/teacher_traj/openmath2-1k-256step

Mini-cache pre-flight (REENTRY_TD_SPEC.md §5.4)::

    ... add --num_samples 50 --shard_size 50 --audit, and use a
    different --output_dir.

What this does:

1. Loads the teacher SFT checkpoint and tokenizer on every rank
   (one process per GPU under torchrun; one process total otherwise).
2. Loads a preprocessed SFT DatasetDict on rank 0 and selects
   ``num_samples`` prompts that fit ``max_prompt_len``. Selection is
   deterministic, so all ranks compute the same sample list.
3. Distributes shards round-robin across ranks: shard ``i`` is owned
   by rank ``i % world_size``. Each rank micro-batches its own shards
   through the deterministic sampler and writes them as memory-mapped
   ``.npy`` files (see :class:`StreamingShardWriter`).
4. ``rank 0`` gathers shard metadata, writes ``manifest.json``, and
   runs the audit if ``--audit`` is set.
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

import dllm  # registers custom configs/models  (per AGENTS.md: import dllm matters)
from dllm.core.samplers.mdlm_deterministic import (
    DeterministicMDLMSampler,
    DeterministicMDLMSamplerConfig,
    sampler_config_hash_payload,
)
from dllm.pipelines.llada_looped.cache_io import (
    CacheManifest,
    ShardInfo,
    StreamingShardWriter,
    TrajectoryCacheConfig,
    _git_commit_hash,
    compute_sampler_config_hash,
    utcnow_iso,
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
                   help="How many prompts to forward at once. Determinism "
                        "within a fixed micro_batch_size is bit-exact, but "
                        "cross-batch-size determinism is not guaranteed "
                        "(FlashAttention etc).")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model load dtype (passed straight through to "
                        "dllm.utils.get_model).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for the teacher forward.")

    # --- Output ---
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--allow_overwrite", action="store_true")

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
        return None, None  # all prompt, no response
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
    shard_dir: Path,
) -> int:
    """Run the sampler on one shard's worth of samples, streaming to disk.

    Returns ``t_shard = max prompt_len + L_resp``. The shard's ``.npy``
    files are pre-allocated as memory-maps inside ``shard_dir`` and
    filled per (sample, step), so peak RAM stays at one micro-batch's
    worth of trajectories instead of the whole shard's fp32 buffer.
    """
    B = len(samples)
    L_resp = sampler_cfg.max_response_len
    K = sampler_cfg.top_k
    num_steps = sampler_cfg.teacher_steps

    max_prompt_len = max(int(s["prompt_ids"].shape[0]) for s in samples)
    t_shard = max_prompt_len + L_resp

    with StreamingShardWriter(
        shard_dir,
        B=B,
        t_shard=t_shard,
        num_steps=num_steps,
        L_resp=L_resp,
        K=K,
        final_sequences_pad=eos_id,
    ) as w:
        for batch_start in range(0, B, micro_batch_size):
            batch_end = min(batch_start + micro_batch_size, B)
            batch = samples[batch_start:batch_end]
            prompts = [torch.as_tensor(s["prompt_ids"], dtype=torch.long) for s in batch]

            out = sampler.sample_with_trajectory(prompts, sampler_cfg)
            seq_np = out.sequences.cpu().numpy()

            for j, s in enumerate(batch):
                i = batch_start + j
                w.set_sample(
                    i,
                    sample_id=int(s["sample_id"]),
                    prompt_len=int(out.prompt_lens[j]),
                    final_sequence=seq_np[j],
                )
                traj = out.trajectories[j]
                if len(traj) != num_steps:
                    raise RuntimeError(
                        f"sample {s['sample_id']}: got {len(traj)} trajectory steps "
                        f"but expected {num_steps}. Did the schedule emit zero-step "
                        "rows? Check teacher_steps vs max_response_len."
                    )
                for step_idx, ts in enumerate(traj):
                    w.set_step(
                        i,
                        step_idx,
                        mask_state=ts.mask_state,
                        top_k_indices=ts.top_k_indices,
                        top_k_logprobs=ts.top_k_logprobs,
                        neg_log_tail_mass=ts.neg_log_tail_mass,
                        top1_conf=ts.top1_conf,
                        entropy=ts.entropy,
                    )
            # Free the micro-batch's trajectory list once it's flushed.
            del out

    return t_shard


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _init_distributed() -> tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)``. Single-process if
    ``LOCAL_RANK`` isn't set in the environment (torchrun / accelerate
    set it; bare python doesn't).
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0

    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank


def _cleanup_distributed() -> None:
    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def main() -> int:
    args = parse_args()
    rank, world_size, local_rank = _init_distributed()

    def log(msg: str, all_ranks: bool = False) -> None:
        if all_ranks:
            print(f"[rank {rank}] {msg}", flush=True)
        elif _is_main(rank):
            print(f"[info] {msg}", flush=True)

    # Every rank checks the same filesystem; they all reach the same
    # decision, so no broadcast is needed and the early-return is safe.
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_overwrite:
        if _is_main(rank):
            print(
                f"[error] output_dir {output_dir} is non-empty. "
                "Pass --allow_overwrite to clobber.",
                file=sys.stderr,
            )
        _cleanup_distributed()
        return 2

    if _is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()  # wait for rank 0 to create output_dir

    # ---- Tokenizer (every rank; cheap) ----
    log(f"loading tokenizer from {args.teacher_ckpt}")
    tokenizer = get_tokenizer(model_name_or_path=args.teacher_ckpt)
    if tokenizer.mask_token_id is None:
        raise RuntimeError(
            "tokenizer has no mask_token_id; LLaDA-style models require it. "
            "Make sure --teacher_ckpt points at a LLaDA SFT checkpoint."
        )

    # ---- Teacher model (one per rank, on its own GPU) ----
    log(f"loading teacher model ({args.dtype}) on cuda:{local_rank}")
    model = get_model(model_name_or_path=args.teacher_ckpt, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if args.device != "cpu" and not next(model.parameters()).is_cuda:
        model = model.to(args.device)

    # ---- Dataset + sample selection (deterministic, every rank gets same list) ----
    log(f"loading dataset from {args.dataset_path} (split={args.dataset_split})")
    ds = load_from_disk(args.dataset_path)
    if args.dataset_split not in ds:
        raise RuntimeError(
            f"split '{args.dataset_split}' missing in dataset; available: {list(ds.keys())}"
        )
    ds_split = ds[args.dataset_split]

    log(f"selecting {args.num_samples} prompts (max_prompt_len={args.max_prompt_len})")
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
        mask_token_id=int(tokenizer.mask_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        bos_token_id=int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else -1,
        vocab_size=int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        tokenizer_name_or_path=args.teacher_ckpt,
    )
    sampler_hash = compute_sampler_config_hash(sampler_config_hash_payload(sampler_cfg))
    git_hash = _git_commit_hash() or "unknown"
    cuda_version = torch.version.cuda  # type: ignore[attr-defined]

    # ---- Plan shards: round-robin across ranks ----
    num_shards = (args.num_samples + args.shard_size - 1) // args.shard_size
    eos_id = int(tokenizer.eos_token_id)

    my_shard_indices = [i for i in range(num_shards) if i % world_size == rank]

    if _is_main(rank):
        log(
            f"generating cache: {args.num_samples} samples, "
            f"{args.teacher_steps} steps, top_k={args.top_k}, "
            f"shard_size={args.shard_size} -> {num_shards} shards, "
            f"distributed across world_size={world_size}"
        )

    # ---- Build my shards ----
    my_shard_infos: list[tuple[int, ShardInfo]] = []
    t_global = time.time()
    for n_done, shard_idx in enumerate(my_shard_indices):
        start = shard_idx * args.shard_size
        end = min(start + args.shard_size, args.num_samples)
        shard_samples = samples[start:end]
        rel_path = f"shard_{shard_idx:04d}"
        shard_path = output_dir / rel_path

        print(
            f"[rank {rank}] shard {shard_idx} "
            f"({n_done + 1}/{len(my_shard_indices)} mine): {len(shard_samples)} samples "
            f"(sample_ids {shard_samples[0]['sample_id']}..{shard_samples[-1]['sample_id']})",
            flush=True,
        )
        t0 = time.time()
        t_shard = build_shard(
            samples=shard_samples,
            sampler=sampler,
            sampler_cfg=sampler_cfg,
            micro_batch_size=args.micro_batch_size,
            eos_id=eos_id,
            shard_dir=shard_path,
        )
        t_total = time.time() - t0
        size_mb = sum(p.stat().st_size for p in shard_path.rglob("*.npy")) / (1 << 20)
        print(
            f"[rank {rank}]   shard {shard_idx}: {size_mb:.1f} MB in {t_total:.1f}s",
            flush=True,
        )

        my_shard_infos.append((
            shard_idx,
            ShardInfo(
                path=rel_path,
                sample_ids=[int(s["sample_id"]) for s in shard_samples],
                t_shard=t_shard,
            ),
        ))

    # ---- Gather shard infos to rank 0 + write manifest ----
    if world_size > 1:
        import torch.distributed as dist
        gathered: list[list[tuple[int, ShardInfo]] | None] = [None] * world_size
        dist.all_gather_object(gathered, my_shard_infos)
        dist.barrier()
        all_infos: list[tuple[int, ShardInfo]] = []
        for lst in gathered:
            all_infos.extend(lst)  # type: ignore[arg-type]
    else:
        all_infos = my_shard_infos

    all_infos.sort(key=lambda x: x[0])
    sorted_shard_infos = [info for _, info in all_infos]

    elapsed = time.time() - t_global

    if _is_main(rank):
        manifest = CacheManifest(
            config=cache_cfg,
            git_commit_hash=git_hash,
            sampler_config_hash=sampler_hash,
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            created_at=utcnow_iso(),
            shards=sorted_shard_infos,
        )
        manifest.save(output_dir)
        log(f"done. {num_shards} shards in {elapsed:.1f}s (wall time)")

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
    _cleanup_distributed()

    # ---- Audit only on rank 0 ----
    if _is_main(rank) and args.audit:
        manifest_for_audit = CacheManifest(
            config=cache_cfg,
            git_commit_hash=git_hash,
            sampler_config_hash=sampler_hash,
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            created_at=utcnow_iso(),
            shards=sorted_shard_infos,
        )
        ok = run_audit(
            output_dir=output_dir,
            manifest=manifest_for_audit,
            min_pass_frac=args.audit_min_pass_frac,
            min_top_k_mass=args.audit_min_top_k_mass,
        )
        if not ok:
            print(
                "[error] audit FAILED. Consider re-running with a larger --top_k, "
                "or relaxing --audit_min_top_k_mass / --audit_min_pass_frac.",
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
    """Audit: in >= min_pass_frac of supervised positions, top-K mass >= min_top_k_mass."""
    from dllm.pipelines.llada_looped.cache_io import read_shard

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
        top_k_mass = np.exp(shard.top_k_logprobs).sum(axis=-1)  # [B, num_steps, L_resp]
        supervised = shard.mask_state
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
