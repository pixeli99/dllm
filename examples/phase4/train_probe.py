"""Phase 4 — step 2: 5-way linear probe from WoDD v_k to AR-CoT token class.

Loads the .pt produced by extract_tape_and_teacher.py and trains a
sklearn LogisticRegression / torch.nn.Linear from v_k to one of:

    0 digit                 — token is fully numeric (after stripping leading
                              whitespace), e.g. "12", "3.14", "-5"
    1 arithmetic operator   — +, -, *, /, =, ×, ÷
    2 variable-binding word — let, set, denote, define, is, be, have
    3 answer-form token     — therefore, so, =, answer, boxed
    4 other                 — everything else (including English connectives)

The pre-registered prediction (WODD_PLAN §3.4, P4-probe):
    probe accuracy > 50 % (chance over 5 classes ≈ 20 %).

Pairing of (v_k, label) is by 'alignment' — for each problem we have
T_step v_k vectors and a teacher trace of L tokens. We use a *position-
proportional* alignment: token i in the trace is paired with v_k where
k = floor(i * T_step / L). This matches the design intuition that later
v_k reflect later reasoning steps in the AR trace; the linear probe
either captures that alignment-conditional signal or it doesn't.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


DIGIT_RE  = re.compile(r"^-?\d+(\.\d+)?$")
OPS       = {"+", "-", "*", "/", "=", "×", "÷"}
VAR_BIND  = {"let", "set", "denote", "define", "is", "be", "have", "suppose"}
ANSWER    = {"therefore", "so", "answer", "boxed", "thus", "hence"}


def token_class(tok: str) -> int:
    """Map a decoded token string to one of 5 classes."""
    s = tok.strip().lower()
    if not s:
        return 4
    if DIGIT_RE.match(s):
        return 0
    if s in OPS:
        return 1
    if s in VAR_BIND:
        return 2
    if s in ANSWER:
        return 3
    return 4


def build_pairs(data, teacher_tok) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct (X, y, k_idx) from the extraction file.

    X: (M, tape_dim) — v vectors flattened over problems and aligned positions
    y: (M,)          — 0..4 class label
    k_idx: (M,)      — which inner step (0..T_step-1) this row came from
    """
    v_all: torch.Tensor = data["v_per_k"]               # (N, T_step, tape_dim)
    traces: List[List[int]] = data["teacher_trace_ids"]
    T_step = int(data.get("T_step", v_all.shape[1]))

    X_rows = []
    y_rows = []
    k_rows = []
    for n in range(v_all.shape[0]):
        L = len(traces[n])
        if L == 0:
            continue
        for i, tok_id in enumerate(traces[n]):
            k = min(T_step - 1, (i * T_step) // L)
            tok_str = teacher_tok.decode([tok_id])
            cls = token_class(tok_str)
            X_rows.append(v_all[n, k].numpy())
            y_rows.append(cls)
            k_rows.append(k)
    return (np.stack(X_rows, 0).astype(np.float32),
            np.array(y_rows, dtype=np.int64),
            np.array(k_rows, dtype=np.int64))


def train_and_eval(X: np.ndarray, y: np.ndarray, k_idx: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    X_tr, X_te, y_tr, y_te, k_tr, k_te = train_test_split(
        X, y, k_idx, test_size=0.2, random_state=seed, stratify=y,
    )
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="multinomial")
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_te)
    acc = accuracy_score(y_te, y_hat)
    chance = max(np.bincount(y_te, minlength=5) / max(1, len(y_te)))
    cm = confusion_matrix(y_te, y_hat, labels=list(range(5))).tolist()

    # Per-k accuracy: does later v_k explain its slice better?
    per_k = {}
    for k in sorted(set(k_te.tolist())):
        mask = (k_te == k)
        per_k[int(k)] = float(accuracy_score(y_te[mask], y_hat[mask])) if mask.any() else None

    return {
        "accuracy":    float(acc),
        "chance":      float(chance),
        "n_train":     int(len(y_tr)),
        "n_test":      int(len(y_te)),
        "class_dist":  np.bincount(y, minlength=5).tolist(),
        "confusion":   cm,
        "per_k_accuracy": per_k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="output of extract_tape_and_teacher.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_path", default=".phase4/probe_report.json")
    args = ap.parse_args()

    data = torch.load(args.data_path, map_location="cpu", weights_only=False)
    teacher_model = data.get("teacher_model", "Qwen/Qwen3-8B-Math")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    teacher_tok = AutoTokenizer.from_pretrained(teacher_model)

    X, y, k_idx = build_pairs(data, teacher_tok)
    print(f"[probe] pairs: {X.shape[0]}, dim={X.shape[1]}, class_dist="
          f"{np.bincount(y, minlength=5).tolist()}", flush=True)

    report = train_and_eval(X, y, k_idx, args.seed)
    print(json.dumps(report, indent=2))

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(report, indent=2))
    print(f"saved report to {args.output_path}")

    pred = "P4-PASS" if report["accuracy"] > 0.5 else "P4-FAIL"
    print(f"{pred}: accuracy={report['accuracy']:.4f} vs threshold 0.5")


if __name__ == "__main__":
    main()
