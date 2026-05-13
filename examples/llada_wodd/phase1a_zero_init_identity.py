"""Phase 1a sanity check: zero-init logit identity.

WODD_PLAN.md §3.1: a WoDD model retrofitted from LLaDA-8B-Base with
T_step=1 and zero_init_wodd=True must produce logits *bit-identical*
to vanilla LLaDA-8B-Base on the same inputs, in fp32, with
max |logit_diff| < 1e-4. If this fails, the drop-in invariant is
broken and training is unsafe -- the cluster-goal kill criterion.

Why this should pass exactly:
  - At zero-init, write_head.content_proj == 0  ->  v_k == 0
  - tape_read.out_proj == 0                     ->  tape_contrib == 0
  - With T_step=1 there is exactly one inner iteration, and the tape
    is empty for k=0 so the read is skipped entirely.
  - The block partition (prelude=8, recurrent=16, coda=8) re-applies
    the same 32 LLaDA-8B blocks in the same order. The recurrent
    block runs ONCE (T_step=1), matching vanilla's single forward pass.
  - The ln_f + ff_out (no weight_tying for LLaDA-8B-Base) projection
    is identical.

We run the comparison on a held-out OpenMath2 batch in fp32 on a
single GPU, loading vanilla and WoDD *sequentially* to fit 8 B params
twice within one H800 80 GB card.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Force offline -- the LLaDA-8B-Base checkpoint is in the local HF cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from datasets import load_from_disk

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
from dllm.pipelines.llada_wodd import LLaDAWoDDConfig, LLaDAWoDDModelLM


DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Base"
DEFAULT_DATA  = ".data/sft/llada/openmath2-500k"


def build_batch(data_path: str, n_examples: int, max_len: int, device: torch.device):
    """Take the first `n_examples` from the OpenMath test split, right-pad
    to max_len, and return cuda tensors (input_ids, attention_mask)."""
    ds = load_from_disk(data_path)
    test = ds["test"]
    # Pick a fixed slice for reproducibility -- not random.
    rows = [test[i] for i in range(n_examples)]
    ids_list   = [row["input_ids"][:max_len] for row in rows]
    pad_id     = 126081  # eos as pad; doesn't matter, masked off by attention_mask.
    T          = max(len(x) for x in ids_list)
    ids_pad    = torch.full((n_examples, T), pad_id, dtype=torch.long)
    mask_pad   = torch.zeros((n_examples, T), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids_pad[i, :len(x)]  = torch.tensor(x, dtype=torch.long)
        mask_pad[i, :len(x)] = 1
    return ids_pad.to(device), mask_pad.to(device), rows


@torch.no_grad()
def forward_vanilla(model_name: str, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Load vanilla LLaDA-8B-Base in fp32, run forward, return logits on CPU."""
    print(f"[vanilla] loading {model_name} in fp32 ...", flush=True)
    t0 = time.time()
    cfg = LLaDAConfig.from_pretrained(model_name, local_files_only=True)
    m = LLaDAModelLM.from_pretrained(
        model_name, config=cfg, torch_dtype=torch.float32, local_files_only=True,
    )
    m = m.to(ids.device).eval()
    print(f"[vanilla] loaded in {time.time()-t0:.1f}s; GPU mem = "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)

    out = m.model.forward(
        input_ids=ids,
        attention_mask=mask,
        attention_bias=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
    )
    logits = out.logits.detach().to(dtype=torch.float32, device="cpu").clone()
    print(f"[vanilla] logits shape: {tuple(logits.shape)}, "
          f"mean={logits.mean().item():.4f}, std={logits.std().item():.4f}",
          flush=True)
    del m, out, cfg
    torch.cuda.empty_cache()
    return logits


@torch.no_grad()
def forward_wodd(model_name: str, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Load WoDD-retrofitted LLaDA-8B in fp32 with T_step_eval=1 and
    zero_init_wodd=True. Run forward, return logits on CPU."""
    print(f"[wodd] retrofitting {model_name} via from_llada_checkpoint "
          f"(t_step_eval=1, zero_init_wodd=True, fp32) ...", flush=True)
    t0 = time.time()
    # NOTE: from_llada_checkpoint forwards **wodd_kwargs straight into the
    # config dict, so we MUST NOT pass HF-loader kwargs like
    # local_files_only here. We rely on HF_HUB_OFFLINE=1 in the env.
    m = LLaDAWoDDModelLM.from_llada_checkpoint(
        model_name,
        torch_dtype=torch.float32,
        prelude_layers=8,
        recurrent_layers=16,
        coda_layers=8,
        t_step_eval=1,
        tape_dim=1024,
        tape_max_writes=4,
        tape_n_heads=8,
        zero_init_wodd=True,
    )
    m = m.to(ids.device).eval()
    print(f"[wodd] loaded in {time.time()-t0:.1f}s; GPU mem = "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)

    out = m(
        input_ids=ids,
        attention_mask=mask,
        T_step=1,
        output_tape_diagnostics=True,
    )
    logits = out.logits.detach().to(dtype=torch.float32, device="cpu").clone()
    gates  = out.write_gates.detach().to(dtype=torch.float32, device="cpu").clone()
    diag   = out.tape_diagnostics
    print(f"[wodd] logits shape: {tuple(logits.shape)}, "
          f"mean={logits.mean().item():.4f}, std={logits.std().item():.4f}",
          flush=True)
    print(f"[wodd] write_gates: shape={tuple(gates.shape)}, "
          f"values={gates.flatten().tolist()}", flush=True)
    if diag is not None:
        print(f"[wodd] tape_diagnostics.T_step = {diag['T_step']}, "
              f"|tape_contrib_rel_norm|={len(diag['tape_contrib_rel_norm'])}",
              flush=True)
    del m, out
    torch.cuda.empty_cache()
    return logits


def compare(logits_a: torch.Tensor, logits_b: torch.Tensor, mask: torch.Tensor, atol: float) -> dict:
    """Report max abs diff, mean abs diff (over non-pad positions only),
    and the worst-position breakdown.

    Note: vanilla LLaDA does logits.mul_(1/sqrt(d_model)) in place when
    scale_logits=True; WoDD does logits = logits * (1.0/sqrt(d_model)).
    Both branches are gated on scale_logits=False for LLaDA-8B-Base, so
    no compensation needed.
    """
    assert logits_a.shape == logits_b.shape, (logits_a.shape, logits_b.shape)
    diff = (logits_a - logits_b).abs()
    # Restrict comparison to non-pad positions.
    valid = mask.cpu().bool().unsqueeze(-1).expand_as(diff)
    diff_valid = diff[valid]
    max_diff   = diff_valid.max().item()
    mean_diff  = diff_valid.mean().item()
    rel_diff   = (diff_valid / logits_a.abs()[valid].clamp_min(1e-12)).mean().item()
    # Top-k worst tokens (over the flattened valid view).
    flat       = diff.flatten()
    topk       = flat.topk(5)
    topk_vals  = topk.values.tolist()
    pass_ = max_diff < atol
    return {
        "max_abs_diff":  max_diff,
        "mean_abs_diff": mean_diff,
        "rel_diff":      rel_diff,
        "top5_diffs":    topk_vals,
        "atol":          atol,
        "pass":          pass_,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--data_path",  default=DEFAULT_DATA)
    ap.add_argument("--n_examples", type=int, default=2)
    ap.add_argument("--max_len",    type=int, default=384,
                    help="truncate input_ids to this length")
    ap.add_argument("--atol",       type=float, default=1e-4)
    ap.add_argument("--device",     default="cuda:0")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available; this Phase 1 check requires a GPU.",
              file=sys.stderr)
        sys.exit(2)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print(f"=== Phase 1a: zero-init logit identity ===", flush=True)
    print(f"  model  : {args.model_name}", flush=True)
    print(f"  data   : {args.data_path}", flush=True)
    print(f"  n_ex   : {args.n_examples}, max_len: {args.max_len}", flush=True)
    print(f"  atol   : {args.atol} (fp32)", flush=True)
    print(f"  device : {device}", flush=True)

    ids, mask, _rows = build_batch(args.data_path, args.n_examples, args.max_len, device)
    print(f"  batch  : input_ids={tuple(ids.shape)}, "
          f"mask sum={int(mask.sum().item())}", flush=True)

    logits_vanilla = forward_vanilla(args.model_name, ids, mask)
    logits_wodd    = forward_wodd(args.model_name, ids, mask)

    rep = compare(logits_wodd, logits_vanilla, mask, args.atol)
    print("=== result ===", flush=True)
    for k, v in rep.items():
        print(f"  {k}: {v}", flush=True)

    if not rep["pass"]:
        print(f"FAIL: max |logit_diff| = {rep['max_abs_diff']:.6e} >= "
              f"{args.atol:.6e} (atol). Zero-init invariant is BROKEN.",
              file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"PASS: max |logit_diff| = {rep['max_abs_diff']:.6e} < "
          f"{args.atol:.6e} (atol). Zero-init invariant holds.", flush=True)
    return rep


if __name__ == "__main__":
    main()
