"""Phase 4 — step 1: extract WoDD tape entries + Qwen3-8B-Math teacher CoT.

For each problem in a held-out GSM8K slice:
  - Run the converged WoDD-8B model on the problem with output_tape_diagnostics=True
    and capture the raw content vectors v_k (BEFORE the gate is applied) for each
    inner step k. WODD_PLAN.md §3.4 calls for probing v_k, not g_k * v_k.
  - Run the Qwen3-8B-Math teacher on the same problem, generate an AR CoT trace,
    record the trace's tokenizations.

We DO NOT train the probe here -- that's `examples/phase4/train_probe.py`.
This separation lets the probe re-train quickly on the same extracted data,
which is the bulk of compute (~ teacher generation × N problems).

The output is a .pt file containing:
  {
    'problem_id': List[str],          # GSM8K id per row
    'v_per_k':    Tensor (N, T_step, tape_dim),
    'teacher_trace_ids': List[List[int]],
    'teacher_trace_text': List[str],
  }

WODD_PLAN.md §3.4 token classes:
  0 — digits (0-9, possibly with sign / decimal)
  1 — arithmetic operators (+, -, *, /, =)
  2 — variable-binding words ("let", "set", "be", "denote", etc)
  3 — answer-form tokens ("therefore", "answer is", "boxed", ...)
  4 — other
The class assignment uses the teacher tokenizer's strings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gsm8k(path: str, n_examples: int):
    if path.startswith("hf:"):
        # e.g. hf:gsm8k:main
        _, name, cfg = path.split(":")
        ds = load_dataset(name, cfg)["test"]
    elif os.path.isdir(path):
        ds = load_from_disk(path)
        if hasattr(ds, "keys") and "test" in ds:
            ds = ds["test"]
    else:
        raise ValueError(f"unrecognised gsm8k path: {path}")
    if n_examples < len(ds):
        ds = ds.select(range(n_examples))
    return ds


@torch.no_grad()
def extract_wodd_tape(wodd_model, tokenizer, problem_text: str, T_step: int,
                      max_len: int = 1024) -> torch.Tensor:
    """Forward `problem_text` through the WoDD model and return the
    pre-gate content vectors v_k for every inner iteration k = 1..T_step.

    We need v_k specifically (not g_k * v_k) because the probe target is
    'what the model would have written'; the gate is a separate decision.
    We pull v_k from the model's internal state via a forward hook on
    write_head.content_proj.
    """
    enc = tokenizer(problem_text, return_tensors="pt", truncation=True,
                    max_length=max_len).to(wodd_model.device)

    v_capture: List[torch.Tensor] = []

    def _hook(_module, _inp, out):
        # out shape: (B, tape_dim).
        v_capture.append(out.detach().to("cpu", dtype=torch.float32))

    h = wodd_model.model.write_head.content_proj.register_forward_hook(_hook)
    try:
        _ = wodd_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            T_step=T_step,
        )
    finally:
        h.remove()

    if not v_capture:
        raise RuntimeError("hook captured nothing; check write_head.content_proj path")
    return torch.stack(v_capture, dim=1).squeeze(0)  # (T_step, tape_dim)


@torch.no_grad()
def teacher_cot(teacher_model, teacher_tok, problem_text: str,
                max_new_tokens: int = 512) -> tuple[List[int], str]:
    """Run AR CoT generation with Qwen3-8B-Math. Returns (token_ids, text)."""
    prompt = (
        "Solve the following math problem. Show your reasoning, then state "
        "the final answer.\n\n"
        f"Problem: {problem_text}\n\nSolution:"
    )
    enc = teacher_tok(prompt, return_tensors="pt").to(teacher_model.device)
    out = teacher_model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=teacher_tok.eos_token_id,
    )
    gen_ids = out[0][enc["input_ids"].shape[1]:].tolist()
    gen_text = teacher_tok.decode(gen_ids, skip_special_tokens=True)
    return gen_ids, gen_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wodd_checkpoint", required=True,
                    help="path to converged WoDD-8B checkpoint")
    ap.add_argument("--teacher_model", default="Qwen/Qwen3-8B-Math")
    ap.add_argument("--gsm8k_path", default="hf:openai/gsm8k:main",
                    help="HF spec or local load_from_disk path")
    ap.add_argument("--n_examples", type=int, default=500)
    ap.add_argument("--T_step", type=int, default=4)
    ap.add_argument("--max_input_len", type=int, default=1024)
    ap.add_argument("--max_teacher_tokens", type=int, default=512)
    ap.add_argument("--output_path",
                    default=".phase4/tape_and_teacher.pt")
    args = ap.parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load WoDD (the bf16 production model).
    from dllm.pipelines.llada_wodd import LLaDAWoDDModelLM
    wodd_tok = AutoTokenizer.from_pretrained(args.wodd_checkpoint)
    wodd_model = LLaDAWoDDModelLM.from_pretrained(
        args.wodd_checkpoint, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()

    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16
    ).to("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0").eval()

    ds = load_gsm8k(args.gsm8k_path, args.n_examples)

    problem_ids: List[str] = []
    v_per_k_list: List[torch.Tensor] = []
    teacher_ids_list: List[List[int]] = []
    teacher_text_list: List[str] = []

    for i, row in enumerate(ds):
        pid = str(row.get("id", i))
        problem_text = row.get("question") or row.get("problem") or row["text"]

        try:
            v = extract_wodd_tape(
                wodd_model, wodd_tok, problem_text,
                T_step=args.T_step, max_len=args.max_input_len,
            )
        except Exception as e:
            print(f"[{i}] WoDD extract failed: {e}", file=sys.stderr)
            continue

        try:
            t_ids, t_txt = teacher_cot(
                teacher_model, teacher_tok, problem_text,
                max_new_tokens=args.max_teacher_tokens,
            )
        except Exception as e:
            print(f"[{i}] teacher gen failed: {e}", file=sys.stderr)
            continue

        problem_ids.append(pid)
        v_per_k_list.append(v)
        teacher_ids_list.append(t_ids)
        teacher_text_list.append(t_txt)

        if i % 25 == 0:
            print(f"[{i}/{args.n_examples}] {pid}: |v|={v.shape}, |trace|={len(t_ids)}",
                  flush=True)

    torch.save({
        "problem_id": problem_ids,
        "v_per_k": torch.stack(v_per_k_list, dim=0),  # (N, T_step, tape_dim)
        "teacher_trace_ids": teacher_ids_list,
        "teacher_trace_text": teacher_text_list,
        "teacher_model": args.teacher_model,
        "wodd_checkpoint": args.wodd_checkpoint,
        "T_step": args.T_step,
    }, args.output_path)
    print(f"saved {len(problem_ids)} rows to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
