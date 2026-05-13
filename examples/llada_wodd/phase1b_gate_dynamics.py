"""Phase 1b sanity check: 100-SGD-step gate dynamics on LLaDA-8B-Base.

WODD_PLAN.md §3.1: run the WoDD trainer for ~100 steps with T_step=2
on a 1K-sample OpenMath2 slice and assert the gate didn't collapse:
  wodd/mean_gate    > 0.05
  wodd/gate_entropy > 0.1

We invoke `examples/llada_wodd/sft.py` as a subprocess via `accelerate
launch --num_processes 1` (single H800), capturing the trainer logs.
After the run we parse `<output_dir>/checkpoint-final/trainer_state.json`
which contains the full log_history including every `wodd/*` metric
the trainer emitted via `Trainer.log`.

Final gate stats are taken as the LAST recorded value (post-step 100)
so we measure where training actually settled, not the init.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SFT  = REPO / "examples/llada_wodd/sft.py"


def run_sft(
    output_dir: str,
    dataset_args: str,
    max_steps: int,
    t_step: int,
    diag_log_every: int,
    batch_size: int,
    seed: int,
    gradient_checkpointing: bool,
    log_file: str,
    write_sparsity_lambda: float | None = None,
    gate_entropy_lambda: float | None = None,
    wodd_lr_mult: float | None = None,
) -> int:
    """Spawn sft.py via accelerate launch on a single GPU.

    Stream the child's stdout/stderr to both our own stdout (live)
    and `log_file` (for post-run parsing). HF Trainer prints log dicts
    as `{'loss': ..., 'wodd/mean_gate': ..., ...}` which we parse later.
    """
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Single-GPU on cuda:0 unless caller already restricted via the env.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cmd = [
        sys.executable, "-u", "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        "--num_machines",  "1",
        "--mixed_precision", "bf16",
        str(SFT),
        "--output_dir", output_dir,
        "--dataset_args", dataset_args,
        "--load_preprocessed_data", "True",
        # Fixed T_step for this sanity check.
        "--t_step_min", str(t_step),
        "--t_step_max", str(t_step),
        "--t_step_eval", str(t_step),
        # Hard stop after 100 steps; num_train_epochs is a soft cap only.
        "--max_steps", str(max_steps),
        "--num_train_epochs", "999",
        # Diagnostics every diag_log_every step; HF logging at the same cadence.
        "--diag_log_every", str(diag_log_every),
        "--logging_steps", str(diag_log_every),
        "--logging_first_step", "True",
        # No checkpoints in the middle; final save only.
        "--save_strategy", "no",
        # The sft.py expects eval set; HF accepts "no" to skip eval.
        # (Note: HF 4.57+ uses --eval_strategy, older versions used
        # --evaluation_strategy. We pass both for safety.)
        "--eval_strategy", "no",
        "--bf16", "True",
        "--gradient_checkpointing", str(gradient_checkpointing),
        "--per_device_train_batch_size", str(batch_size),
        "--per_device_eval_batch_size",  str(batch_size),
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "2e-5",
        "--warmup_ratio", "0.0",
        "--seed", str(seed),
        "--report_to", "none",
        # HF default strips columns not used by the model.forward signature
        # (`messages`, `prompt_len`) so the collator sees only input_ids / labels.
        "--remove_unused_columns", "True",
    ]
    if write_sparsity_lambda is not None:
        cmd += ["--write_sparsity_lambda", str(write_sparsity_lambda)]
    if gate_entropy_lambda is not None:
        cmd += ["--gate_entropy_lambda", str(gate_entropy_lambda)]
    if wodd_lr_mult is not None:
        cmd += ["--wodd_lr_mult", str(wodd_lr_mult)]
    print("[phase1b] launching:", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log_file, "w", buffering=1) as fout:
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(REPO),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fout.write(line)
        rc = proc.wait()
    print(f"[phase1b] sft.py exited rc={rc} after "
          f"{time.time()-t0:.1f}s", flush=True)
    return rc


def parse_log_history(log_file: str) -> list[dict]:
    """Pull log dicts out of an sft.py stdout capture.

    HF Trainer prints each `Trainer.log(...)` payload as a Python dict
    repr on its own line, e.g.:
        {'loss': 0.0086, 'grad_norm': 0.037..., 'epoch': 0.0}
        {'wodd/T_step': 2.0, 'wodd/mean_gate': 0.0815, ...}

    We capture every `{ ... }`-only line that parses as a dict via
    ast.literal_eval and add a synthetic `step` field counting log
    entries within each training step (the trainer logs both a `loss`
    line and a `wodd/*` line per logging interval; we synthesize a step
    counter so downstream sorting is stable).

    Robustness: the trainer doesn't print explicit step indices in
    every payload, but it does include `epoch`. We use lexical order
    in the file as the step ordering since logs are emitted in-order.
    """
    import ast
    import re

    entries: list[dict] = []
    # Lines that *start* with '{' and *end* with '}' on the same line,
    # possibly with a tqdm progress-bar prefix. We strip everything
    # before the first '{'.
    DICT_LINE = re.compile(r"\{.*\}\s*$")
    seen = 0
    with open(log_file, "r") as fin:
        for raw in fin:
            line = raw.rstrip("\n")
            # Strip ANSI escape codes from tqdm.
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
            # Take only the trailing dict, skipping any tqdm carriage-return
            # progress bar prefix.
            m = DICT_LINE.search(line)
            if not m:
                continue
            payload = m.group(0).strip()
            try:
                d = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(d, dict):
                continue
            seen += 1
            d.setdefault("step", float(seen))
            entries.append(d)
    return entries


def extract_wodd_series(log_history: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """Group all wodd/* metrics by key, each as a list of (step, value)."""
    series: dict[str, list[tuple[float, float]]] = {}
    for entry in log_history:
        step = entry.get("step", -1)
        for k, v in entry.items():
            if not isinstance(k, str) or not k.startswith("wodd/"):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            series.setdefault(k, []).append((float(step), fv))
    for k in series:
        series[k].sort()
    return series


def report(series: dict[str, list[tuple[float, float]]],
           mean_gate_min: float,
           gate_entropy_min: float) -> dict:
    out = {"per_metric_final": {}, "per_metric_first": {}, "step_count": {}}
    for k, ts in series.items():
        out["per_metric_final"][k] = ts[-1] if ts else None
        out["per_metric_first"][k] = ts[0] if ts else None
        out["step_count"][k] = len(ts)

    # Use the last reported value for each metric as the "settled" value.
    mg = series.get("wodd/mean_gate", [(0, 0.0)])[-1][1]
    ge = series.get("wodd/gate_entropy", [(0, 0.0)])[-1][1]
    out["mean_gate_final"] = mg
    out["gate_entropy_final"] = ge
    out["thresholds"] = {
        "mean_gate_min": mean_gate_min,
        "gate_entropy_min": gate_entropy_min,
    }
    out["pass_mean_gate"]    = mg > mean_gate_min
    out["pass_gate_entropy"] = ge > gate_entropy_min
    out["pass"] = bool(out["pass_mean_gate"] and out["pass_gate_entropy"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir",
                    default=".models/wodd-phase1b")
    ap.add_argument("--dataset_args",
                    default=".data/sft/llada/openmath2-500k[train:1000,test:100]")
    ap.add_argument("--max_steps",         type=int,   default=100)
    ap.add_argument("--t_step",            type=int,   default=2)
    ap.add_argument("--diag_log_every",    type=int,   default=10)
    ap.add_argument("--batch_size",        type=int,   default=1)
    ap.add_argument("--seed",              type=int,   default=0)
    ap.add_argument("--mean_gate_min",     type=float, default=0.05)
    ap.add_argument("--gate_entropy_min",  type=float, default=0.1)
    ap.add_argument("--skip_train", action="store_true",
                    help="just re-parse an existing run")
    ap.add_argument("--no_gc",      action="store_true",
                    help="disable gradient checkpointing (faster, more mem)")
    # Optional overrides for the loss-coefficient sweep recommended by
    # WODD_PLAN.md §5 (Risk C) when the gate collapses under defaults.
    ap.add_argument("--write_sparsity_lambda", type=float, default=None)
    ap.add_argument("--gate_entropy_lambda",   type=float, default=None)
    ap.add_argument("--wodd_lr_mult",          type=float, default=None)
    args = ap.parse_args()

    print(f"=== Phase 1b: 100-step gate dynamics ===", flush=True)
    print(f"  output_dir       : {args.output_dir}", flush=True)
    print(f"  dataset_args     : {args.dataset_args}", flush=True)
    print(f"  max_steps        : {args.max_steps}", flush=True)
    print(f"  t_step (fixed)   : {args.t_step}", flush=True)
    print(f"  diag_log_every   : {args.diag_log_every}", flush=True)
    print(f"  batch_size       : {args.batch_size}", flush=True)
    print(f"  thresholds       : mean_gate > {args.mean_gate_min}, "
          f"gate_entropy > {args.gate_entropy_min}", flush=True)

    log_file = str(Path(args.output_dir) / "phase1b.log")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        rc = run_sft(
            output_dir=args.output_dir,
            dataset_args=args.dataset_args,
            max_steps=args.max_steps,
            t_step=args.t_step,
            diag_log_every=args.diag_log_every,
            batch_size=args.batch_size,
            seed=args.seed,
            gradient_checkpointing=(not args.no_gc),
            log_file=log_file,
            write_sparsity_lambda=args.write_sparsity_lambda,
            gate_entropy_lambda=args.gate_entropy_lambda,
            wodd_lr_mult=args.wodd_lr_mult,
        )
        if rc != 0:
            print(f"ERROR: sft.py returned rc={rc}", file=sys.stderr, flush=True)
            sys.exit(rc)

    log_history = parse_log_history(log_file)
    print(f"[phase1b] log_history entries: {len(log_history)}", flush=True)
    series = extract_wodd_series(log_history)
    rep    = report(series, args.mean_gate_min, args.gate_entropy_min)

    print("=== wodd metric trajectories ===", flush=True)
    for k in sorted(series.keys()):
        ts = series[k]
        first = ts[0]
        last  = ts[-1]
        # Show first, last, and a middle sample.
        mid   = ts[len(ts) // 2]
        print(f"  {k:40s}  first(step={first[0]:.0f})={first[1]:+.5f}"
              f"  mid(step={mid[0]:.0f})={mid[1]:+.5f}"
              f"  last(step={last[0]:.0f})={last[1]:+.5f}"
              f"  n={len(ts)}", flush=True)

    print("=== gate-collapse check ===", flush=True)
    print(f"  wodd/mean_gate    final: {rep['mean_gate_final']:.6f}   "
          f"(min: {rep['thresholds']['mean_gate_min']})   "
          f"{'PASS' if rep['pass_mean_gate'] else 'FAIL'}", flush=True)
    print(f"  wodd/gate_entropy final: {rep['gate_entropy_final']:.6f}   "
          f"(min: {rep['thresholds']['gate_entropy_min']})   "
          f"{'PASS' if rep['pass_gate_entropy'] else 'FAIL'}", flush=True)

    Path("phase1b_report.json").write_text(json.dumps(rep, indent=2))
    if not rep["pass"]:
        print("Phase 1b FAILED — gate collapsed.", file=sys.stderr, flush=True)
        sys.exit(1)
    print("Phase 1b PASS — gate is alive.", flush=True)


if __name__ == "__main__":
    main()
