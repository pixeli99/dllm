"""Phase 1 sanity: zero-init WoDD drop-in must equal frozen LLaDA-8B-Base.

We load `LLaDA-8B-Base`, build LLaDAWoDDModelLM via from_llada_checkpoint
with zero_init_wodd=True, and verify on a held-out OpenMath sample that
with T_step=1 the MDLM logits match the vanilla LLaDAModelLM logits to
within fp32 1e-4 absolute.

This is a PRE-TRAINING sanity check. If it fails, training is unsafe --
the zero-init invariant is broken at GPU scale (likely a dtype /
normalization issue not caught by CPU unit tests). Fix before Phase 2.

Run (single GPU is enough):
    python examples/llada_wodd/phase1_sanity.py \
        --model_name_or_path GSAI-ML/LLaDA-8B-Base \
        --n_samples 64 \
        --tolerance 1e-4
"""

from __future__ import annotations

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", default="GSAI-ML/LLaDA-8B-Base"
    )
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument(
        "--seq_len", type=int, default=512,
        help="Truncate inputs to this length for the comparison."
    )
    parser.add_argument(
        "--dataset", default="nvidia/OpenMathInstruct-2",
        help="Held-out source; we just pull a few text samples."
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.llada_wodd import LLaDAWoDDModelLM
    from dllm.utils import get_tokenizer

    print(f"[phase1] loading vanilla LLaDA from {args.model_name_or_path}")
    vanilla = LLaDAModelLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype
    ).to(args.device).eval()

    print("[phase1] retrofitting into WoDD with zero_init_wodd=True")
    wodd = LLaDAWoDDModelLM.from_llada_checkpoint(
        args.model_name_or_path,
        torch_dtype=dtype,
        prelude_layers=8,
        recurrent_layers=16,
        coda_layers=8,
        tape_dim=1024,
        tape_max_writes=16,
        tape_n_heads=8,
        t_step_eval=1,        # critical: T_step=1 for the strict drop-in check
        zero_init_wodd=True,
    ).to(args.device).eval()

    tok = get_tokenizer_compat(args.model_name_or_path)

    # Pull a tiny held-out sample from the dataset.
    samples = load_eval_samples(args.dataset, args.n_samples, args.seq_len, tok)
    print(f"[phase1] comparing logits on {len(samples)} samples, "
          f"max_diff tolerance={args.tolerance}")

    max_diff = 0.0
    n_failed = 0
    for i, (input_ids, attention_mask) in enumerate(samples):
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)

        with torch.no_grad():
            out_v = vanilla(
                input_ids=input_ids, attention_mask=attention_mask,
                return_dict=True,
            )
            out_w = wodd(
                input_ids=input_ids, attention_mask=attention_mask,
                T_step=1, return_dict=True,
            )

        diff = (out_v.logits.float() - out_w.logits.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > args.tolerance:
            n_failed += 1
            print(f"  [WARN] sample {i}: max_diff={diff:.6e} > tol")

    print(f"\n[phase1] DONE. max_diff over {len(samples)} samples = {max_diff:.6e}")
    print(f"[phase1] tolerance: {args.tolerance}")
    if max_diff <= args.tolerance:
        print("[phase1] PASS — zero-init drop-in holds at GPU scale.")
    else:
        print(f"[phase1] FAIL — {n_failed}/{len(samples)} samples exceeded "
              f"tolerance. DO NOT proceed to Phase 2 training.")
        raise SystemExit(1)


def get_tokenizer_compat(model_name_or_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )


def load_eval_samples(dataset: str, n: int, seq_len: int, tok):
    """Yield up to `n` (input_ids, attention_mask) pairs for the comparison.

    We don't run the MDLM loss here — only the forward logits — so we just
    need any plausible token sequences. We use the first `n` test problems
    from the OpenMath split, tokenized + truncated.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, split=f"train[:{n}]")
    samples = []
    for ex in ds:
        text = ex.get("problem") or ex.get("question") or ex.get("text") or str(ex)
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        samples.append((enc["input_ids"], enc["attention_mask"]))
    return samples


if __name__ == "__main__":
    main()
