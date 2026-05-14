"""Phase 6 falsification (c): 2-digit addition.

Pre-registered prediction (WODD_PLAN §3.6, P6-falsification (c)):
  WoDD ≈ MetaState ≈ baseline below the bit-budget threshold;
  WoDD ≈ MetaState above the bit-budget threshold (where extra
  capacity helps). Crucially, 2-digit add (a + b, both in [0,99])
  needs ~ 14 bits of working memory, well below any reasonable
  bit budget, so the prediction is that ALL three arms are at the
  same plateau and WoDD's append-only design provides no measurable
  advantage. If WoDD does win here, the "bit-budget threshold"
  framing is wrong.

We do NOT train on 2-digit add; we test the converged Phase 2/3
models (vanilla / metastate / wodd) zero-shot.

Decoding: greedy unmasking via the MDLM sampler at the inference
length matching "%d + %d =" suffix. We pass the same sampler
config across all three arms so the comparison is at matched FLOPs.

Output (per arm): a JSON with raw correctness array, accuracy,
problem-difficulty breakdown (by sum magnitude).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer


ARM_MODEL_LM = {
    "vanilla":   ("dllm.pipelines.llada.models.modeling_llada",   "LLaDAModelLM"),
    "latts":     ("dllm.pipelines.llada_latts",                   "LLaDALATTSModelLM"),
    "metastate": ("dllm.pipelines.llada_metastate",               "LLaDAMetaStateModelLM"),
    "dpad":      ("dllm.pipelines.llada_dpad",                    "LLaDADPadModelLM"),
    "wodd":      ("dllm.pipelines.llada_wodd",                        "LLaDAWoDDModelLM"),
}


def gen_problems(n: int, seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        out.append((a, b, a + b))
    return out


def load_model(arm: str, ckpt: str):
    import importlib
    modpath, clsname = ARM_MODEL_LM[arm]
    if modpath.startswith("."):
        modpath = "dllm" + modpath
    mod = importlib.import_module(modpath)
    cls = getattr(mod, clsname)
    return cls.from_pretrained(ckpt, torch_dtype=torch.bfloat16)


@torch.no_grad()
def score_problems(model, tokenizer, problems, max_new_tokens=8) -> list[int]:
    """Return a list of 0/1 correctness per problem.

    For each problem we feed a prompt "What is A + B? Answer: " and
    ask the model to fill in. We greedy-decode using a simple
    block-fill approach (consistent with the LLaDA sampler family).
    """
    from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_cfg = MDLMSamplerConfig(
        max_new_tokens=max_new_tokens,
        steps=max_new_tokens,
        block_size=max_new_tokens,
        cfg_scale=0.0,
    )

    correct = []
    for a, b, gt in problems:
        prompt = f"What is {a} + {b}? Answer: "
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)[0]
        gen = sampler.sample(inputs=[ids], config=sampler_cfg, return_dict=False)
        text = tokenizer.decode(gen[0][len(ids):], skip_special_tokens=True)
        # extract first integer
        digits = "".join(ch for ch in text.split()[:1][0] if ch.isdigit()) if text.split() else ""
        try:
            pred = int(digits) if digits else None
        except ValueError:
            pred = None
        correct.append(1 if pred == gt else 0)
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_MODEL_LM.keys()))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_problems", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_path", default=None)
    args = ap.parse_args()

    if args.output_path is None:
        args.output_path = f".eval/phase6/two_digit_add/{args.arm}_seed{args.seed}.json"
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = load_model(args.arm, args.checkpoint).to("cuda").eval()
    problems = gen_problems(args.n_problems, args.seed)
    correct = score_problems(model, tok, problems)

    # Difficulty bucketing: smaller sums are easier (no carry).
    buckets = {"sum<=18": [], "19<=sum<=99": [], "sum>=100": []}
    for (a, b, gt), c in zip(problems, correct):
        if gt <= 18:                buckets["sum<=18"].append(c)
        elif gt <= 99:              buckets["19<=sum<=99"].append(c)
        else:                       buckets["sum>=100"].append(c)
    bucket_acc = {k: (sum(v) / len(v) if v else None) for k, v in buckets.items()}

    report = {
        "arm": args.arm,
        "checkpoint": args.checkpoint,
        "n_problems": args.n_problems,
        "seed": args.seed,
        "accuracy": sum(correct) / len(correct),
        "bucket_accuracy": bucket_acc,
    }
    print(json.dumps(report, indent=2))
    Path(args.output_path).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
