# WoDD Experimental Results

This file records every numerical result from the 6-phase WoDD program
defined in `WODD_PLAN.md` and `WODD_GOAL.md`. Sections are appended
chronologically. Null and negative results are first-class.

Format per entry:
- exact code commit + config
- raw numbers (mean ± std over seeds, no rounding)
- 1-paragraph interpretation tagging which pre-registered prediction the
  result confirms or falsifies
- any deviation from the plan, with the reason

---

## Pre-run housekeeping (2026-05-13)

### Test-suite status

Branch: `lpx/wodd` @ `7581b83` (parent: `31d45c3 docs: add WODD_GOAL.md`).
Command: `pytest tests/wodd/`.

Result: **27 passed, 0 failed, 0 skipped** (was 21 pass + 6 skip in the
CPU sandbox).

The 6 previously-skipped forward tests now run with `lm_eval` /
`transformers` / CUDA visible. Two latent bugs surfaced and were patched:

1. **Upstream LLaDA bug (not WoDD)**:
   `dllm/pipelines/llada/models/modeling_llada.py:1456` hard-codes
   `model_config.init_device = "cuda"` in `LLaDAModelLM.__init__`,
   overriding the user-set value (`init_device='cpu'` in the toy
   config). With CUDA visible, the toy model ends up on `cuda:0`
   regardless. We fix the toy test by reading
   `next(m.parameters()).device` and placing inputs to match. The
   upstream override is left alone to limit blast radius; a separate
   PR should restore the documented "init on CPU then move" semantics.
2. **Toy-config activation mismatch**:
   `LLaDALlamaBlock` implements SwiGLU gating *manually* (separate
   `ff_proj` + `up_proj`, elementwise multiply). It needs an
   activation with `output_multiplier == 1` (SiLU). The toy cfg
   inherited the default `activation_type=swiglu` whose 0.5
   multiplier halved one branch and broke the elementwise multiply.
   Fix: explicit `activation_type='silu'` in the toy config.

Neither bug affects the production WoDD code path or the planned
Phase 1–6 runs; both are toy-test-only.

### Cluster environment

8× NVIDIA H800 (80 GB each), CUDA 13.0, driver 580.65.06.
`torch==2.6.0+cu124`, `transformers==4.57.0`, `accelerate==1.11.0`,
`deepspeed==0.18.0`, `datasets==4.8.4`, `lm_eval==0.4.9.1`.

---

## Phase 1 — sanity (2026-05-13)

### Phase 1a: zero-init logit identity on LLaDA-8B-Base

**Status: PASS (bit-identical, 0.0 abs diff).** Cluster-goal kill criterion
holds at GPU scale.

- Commit: `lpx/wodd@1fd8bb4` (`examples/llada_wodd: add Phase 1a zero-init identity script`)
- Script: `examples/llada_wodd/phase1a_zero_init_identity.py`
- Compute: 1× H800 80 GB, fp32 throughout.
- Model: `GSAI-ML/LLaDA-8B-Base` (n_layers=32, d_model=4096, embedding_size=126464,
  activation_type=silu, weight_tying=False, max_seq_len=4096), local HF cache.
- Data: first 2 examples from `.data/sft/llada/openmath2-500k` test split,
  right-truncated to T=384. mask sum = 768.
- WoDD config: `prelude_layers=8, recurrent_layers=16, coda_layers=8,
  t_step_eval=1, tape_dim=1024, tape_max_writes=4, tape_n_heads=8,
  zero_init_wodd=True`.
- Method: load LLaDA-8B-Base in fp32 → forward → save logits;
  free; load `LLaDAWoDDModelLM.from_llada_checkpoint(..., torch_dtype=fp32,
  t_step_eval=1, zero_init_wodd=True)` → forward(T_step=1) → compare.

**Raw numbers**

| metric                      | value |
|----------------------------|------:|
| max abs logit diff          | **0.000000e+00** |
| mean abs logit diff         | 0.000000e+00 |
| relative diff               | 0.0 |
| top-5 worst per-token diff  | [0.0, 0.0, 0.0, 0.0, 0.0] |
| atol (kill threshold)       | 1e-4 |

**Side observations.**
* `write_gates` at init: 0.16336, 0.17209 (per batch element). Above the
  bias-init floor sigmoid(−2)≈0.119 because the gate_proj weights get the
  default Kaiming init that adds a small positive drift on top of the bias;
  not a concern.
* `tape_diagnostics.tape_contrib_rel_norm` is an empty list. Correct: at
  T_step=1 there is no second iteration, so the cross-attn read is never
  invoked (tape is empty for k=0).

**Interpretation.** Confirms WODD_PLAN.md §3.1 prediction: drop-in invariant
holds in fp32 to floating-point exactness, not just approximately. The
T_step=1 / zero_init pathway truly degenerates to vanilla LLaDA. Pre-
registered kill criterion does not fire; Phase 2 is unblocked.

### Phase 1b: 100-SGD-step gate dynamics

**Status: defaults FAIL the gate-non-collapse assertion; recommended
mitigation (100× higher `gate_entropy_lambda`) PASSES cleanly.**
This is a precise replication of the failure signature WODD_PLAN.md §7
predicts (`mean_gate` near 0 — collapse; increase entropy reg) and
indicates the production defaults need adjustment before Phase 2.

- Commit: `lpx/wodd@64e4c2b`
  (`examples/llada_wodd: add Phase 1b gate-dynamics runner; dedupe t_step_eval`)
- Script: `examples/llada_wodd/phase1b_gate_dynamics.py`
- Compute: 1× H800 80 GB, bf16 mixed precision via accelerate launch
  `--num_processes 1 --mixed_precision bf16`, gradient checkpointing on.
- Backbone: frozen `LLaDA-8B-Base`. Trainable surface ≈ **14.73 M / 8.03 B
  params (0.18%)** — write_head + tape_read only.
- Data: `.data/sft/llada/openmath2-500k[train:1000,test:100]`.
- Optimizer: AdamW, base LR 2e-5, no warmup, cosine over 100 steps decays
  to ~0 by step 100. `wodd_lr_mult=10` → WoDD modules learn at 2e-4.
- T_step (fixed): 2. `diag_log_every=10`. `seed=0`.

**Three configurations, 100 steps each:**

| run | B | sparsity_λ | entropy_λ | lr_mult | final mean_gate | final entropy | final loss_mdlm | tape_contrib | verdict |
|---|---|---|---|---|---|---|---|---|---|
| defaults     | 1 | 1e-2 | 1e-3 | 10 | 0.02136 | 0.10320 | 4.288 | 0.2355 | FAIL |
| defaults     | 8 | 1e-2 | 1e-3 | 10 | **0.01238** | **0.06564** | 0.6263 | 0.5560 | **FAIL** |
| ent×100      | 8 | 1e-2 | 1e-1 | 10 | **0.49500** | **0.68979** | 0.6195 | 0.5675 | **PASS** |

**Default trajectory (B=8, 100 steps).** `mean_gate` monotonically decays
from 0.0881 (step 10) → 0.0154 (step 60) → 0.0124 (step 100). Entropy
mirrors this: 0.298 → 0.079 → 0.066. The sparsity penalty (1e-2·mean(g))
dominates the early-training loss landscape before the model has had time
to learn that tape reads are informative; the gradient pulls gate_proj.bias
toward more negative values, the gate collapses, and the model essentially
trains the tape_read.out_proj on a near-zero-content tape.

**ent×100 trajectory (B=8, 100 steps).** With `gate_entropy_lambda=0.1`,
`mean_gate` rises monotonically from 0.0881 → 0.46 → 0.495 and entropy
converges to 0.690 ≈ ln(2) (max). The gate lands near 0.5 (uninformative
but "alive"), which is the expected behaviour when entropy regularisation
dominates: no preference, but the modules are receiving useful gradient.

**Interpretation.** WODD_PLAN.md §3.1 expects `mean_gate > 0.05` and
`gate_entropy > 0.1` after 100 steps under the trainer's defaults. The
default `gate_entropy_lambda=1e-3` (10× weaker than `write_sparsity_lambda`)
is insufficient to keep the gate alive at this scale — the failure
signature predicted in §7 ("near 0 — collapse; increase entropy reg") is
exactly what we see. The 100× higher entropy reg recommended in §5
(Risk C: "anneal λ_entropy from high to low") restores the dynamics.
None of this falsifies any of the pre-registered P2–P6 predictions; it
does suggest the production `scripts/llada_wodd/run.sh` defaults should
use `gate_entropy_lambda ≥ 1e-2` (and ideally an annealing schedule
high→low) for Phase 2.

**Deviation from plan.** The plan said B=2 ("e.g., 1 K samples"); we ran
at B=8 to reduce single-batch loss variance (B=1 produced `loss_mdlm`
spikes 0.008→4.29 due to MDLM timestep `t` occasionally masking ~0 tokens
at low `t`). Both B=1 and B=8 default runs gave qualitatively the same
gate-collapse signature, so the deviation does not change the conclusion.

**Action items into Phase 2.**
1. Bump `gate_entropy_lambda` to at least 1e-2 in
   `scripts/llada_wodd/run.sh` and `MDLMWoDDConfig` defaults, OR
   implement an entropy-anneal schedule (e.g., 1e-1 → 1e-3 over the first
   10% of steps).
2. If Phase 2 sees gate collapse again at 8×H800 effective batch (16),
   apply the same mitigation — but report the gap honestly first.

---

## Phase 2 — infrastructure ready, training pending (2026-05-13)

This is a status entry, not a result. Phase 2 actual training (5 arms ×
5 seeds × ~6–10 GPU-h each ≈ 150–250 GPU-h) is the next operator action.

### Baselines implemented and unit-tested

Commit `lpx/wodd@39b6111` (`pipelines: add LATTS / MetaState / DPad
baselines for Phase 2`). Three new pipelines:

| Arm | Pipeline | Trainable surface (toy d=64) | Strict drop-in invariant at zero-init |
|-----|----------|------------------------------|---------------------------------------|
| A0 | (vanilla `LLaDAModelLM`)             | 0 (frozen)             | — |
| A1 | `dllm.pipelines.llada_latts`         | 0 (frozen)             | yes (T_step=1, max diff 0.0) |
| A2 | `dllm.pipelines.llada_metastate`     | state_init + state_read + state_update | yes (zero_init_metastate=True, max diff 0.0) |
| A3 | `dllm.pipelines.llada_dpad`          | suffix scratchpad      | **no** — softmax renormalisation over K extra positions causes O(K/T) drift even with K/V at scratch positions = 0; documented and bounded by test |
| A4 | `dllm.pipelines.llada_wodd` (existing)| write_head + tape_read | yes (zero_init_wodd=True, max diff 0.0 per Phase 1a) |

Unit tests under `tests/baselines/test_baselines_forward.py` (10 cases) +
the existing `tests/wodd/` (27 cases). All **37/37 pass** at `lpx/wodd@23fcd38`.

### Param counts at production scale (8 B backbone, 1 H800 fp32 load + bf16 train)

Empirical from a 2-step Phase 2 smoke run (`metastate` arm, default
`state_slots=64`, `state_dim=1024`, `state_n_heads=8`):

| Arm | Trainable params | % of backbone |
|-----|-----------------:|--------------:|
| A2 MetaState (slots=64, dim=1024) | 26 301 440 | 0.327 % |
| A4 WoDD (tape_dim=1024, max_writes=16) | 14 727 169 | 0.183 % |

The MetaState slot-count is sized for Phase 3's capacity-sweep envelope
(M ∈ {64,128,256,512,1024} → state_init param count scales linearly with
M while StateRead / StateUpdate stay constant; test
`test_metastate_state_param_count_independent_of_slots` pins this).
A1 LATTS and A0 vanilla have no trainable adapter; A3 DPad's count at
`dpad_len=512` is `512*4096 = 2.1 M` (~0.026 %) — well below the WODD_PLAN
"~1 %" rough target, which would need an extra MLP head on the scratchpad
or larger `dpad_len`. We flag this gap rather than tuning to hit the
exact %; matched FLOPs is the explicit target, not matched parameter
count, per WODD_PLAN.md §3.2 footnote.

### Phase 2 launcher

Commit `lpx/wodd@23fcd38` (`phase2: unified train entry + launcher
scripts (5-arm head-to-head)`). Single entry point dispatches all five
arms by `--arm` flag and freezes the LLaDA backbone for every arm except
vanilla. Smoke-tested end-to-end on all five arms (single H800, B=1,
max_steps=1–2) — checkpoint-final written for each.

Run a single cell:
```
bash scripts/phase2/launch_arm.sh wodd 0       # arm=wodd, seed=0
```
Run the full grid:
```
bash scripts/phase2/launch_all.sh              # 5 arms × 5 seeds
```

The wodd cell pins `--gate_entropy_lambda 1e-2 --write_sparsity_lambda
1e-2` (Phase 1b mitigation; the trainer default 1e-3 collapsed the gate
within 100 steps).

### Not yet implemented in this session

- Phase 2 / 3 / 4 / 6 *training* runs and *evaluation* — all the
  scaffolding is in place (see below); compute remains to be spent.
- Phase 5 Dream-7B port of the WoDD architecture.

---

## Phase 2–6 launcher + eval infrastructure (2026-05-14)

Commit `lpx/wodd@0233ee1` (`phase 2-6 infrastructure: per-arm eval
entries + launchers`). The full 6-phase program is now runnable from
the shell with no further coding; only compute and human go/no-go
remain. 37/37 tests still pass.

### Per-arm eval registrations

Each new pipeline ships an `eval.py` that registers itself with
`lm_eval`'s `--model` namespace and with `transformers.AutoConfig` /
`AutoModel`. After this commit, `lm-eval --model X --model_args
pretrained=PATH` works for every arm:

| Arm | `lm_eval --model` flag | HF `model_type` |
|-----|------------------------|-----------------|
| A0 vanilla    | `llada`           | `llada`           |
| A1 LATTS      | `llada_latts`     | `llada_latts`     |
| A2 MetaState  | `llada_metastate` | `llada_metastate` |
| A3 DPad       | `llada_dpad`      | `llada_dpad`      |
| A4 WoDD       | `llada_wodd`      | `llada_wodd`      |

Smoke-tested: importing each `eval.py` populates `lm_eval.MODEL_REGISTRY`
and `AutoConfig.for_model("llada_X")` resolves without
`trust_remote_code`.

### Phase 2 launcher and eval

```
scripts/phase2/launch_arm.sh ARM SEED          # one (arm, seed) cell
scripts/phase2/launch_all.sh                   # full 5×5 grid
scripts/phase2/eval_arm.sh ARM CHECKPOINT [num_gpu]
                                               # gsm8k_cot + minerva_math
                                               # + humaneval + mbpp
```
Greedy decoding (cfg_scale=0.0) for pass@1 per WODD_PLAN.md §3.2.
Self-consistency N=16 is a separate run pattern (not in this script).

### Phase 3 (headline scaling-law) launcher and eval

```
scripts/phase3/launch_capacity_sweep.sh        # 10 cells (5 + 5)
scripts/phase3/eval_sweep.sh                   # GSM8K + MATH-500 only
```
Sweeps exactly the §3.3 specification:
* MetaState slots ∈ {64, 128, 256, 512, 1024} × dim 1024.
* WoDD (tape_dim, T_step) ∈ {(256,4),(512,4),(1024,4),(1024,8),(1024,16)}.

### Phase 4 linear probe (needs a converged WoDD checkpoint)

```
python examples/phase4/extract_tape_and_teacher.py \
    --wodd_checkpoint .models/phase3/wodd_d1024_T16/checkpoint-final \
    --teacher_model Qwen/Qwen3-8B-Math \
    --gsm8k_path hf:openai/gsm8k:main \
    --n_examples 500
python examples/phase4/train_probe.py --data_path .phase4/tape_and_teacher.pt
```
The pre-gate `v_k` are captured via a forward hook on
`write_head.content_proj`. Token classes (5-way) from
WODD_PLAN.md §3.4: digit / operator / variable-binding / answer-form /
other. P4-PASS: accuracy > 0.5 (chance ≈ 0.2).

### Phase 6 falsification

```
scripts/phase6/eval_falsification.sh           # all three settings
```
* (a) GLUE cola, short-output classification — predicted WoDD ≈ vanilla
* (b) TriviaQA closed-book — predicted WoDD ≈ vanilla
* (c) 2-digit addition (synthetic) via
  `examples/phase6/eval_two_digit_add.py` — predicted
  WoDD ≈ MetaState below the bit-budget threshold; separates above.

Per the plan only the three named arms (vanilla / metastate / wodd)
are in the §3.6 battery; LATTS and DPad are out of scope for Phase 6.

### What's still missing

* **Phase 5 Dream-7B port**: needs a `dllm/pipelines/dream_wodd/`
  pipeline analogous to `llada_wodd`, plus the matching eval entry.
  Not started; flagged for the next session.
* **GPU compute**: every step from Phase 2 onward needs cluster time.
  Total budget for the full 6-phase program: roughly 200-300 GPU-h on
  8×H800 (mostly Phase 2 and Phase 3 training).
* **`git push` access**: cluster proxy on port 20172 tunnels HTTPS
  cleanly; the push fails because no GitHub credential is configured
  (no `GH_TOKEN` env, no `~/.netrc`, no `gh` CLI). The operator needs
  to supply a PAT or arrange SSH for the local commits to reach origin.


---

## Phase 2 progress — WoDD seed=0 pilot (2026-05-14)

**Status**: training **done**; eval **in-flight** (numerical scores not
yet landed). Posting per the plan's "push intermediate results after
each phase; do not wait for completion" rule. This is the first cell
of the §3.2 grid. The remaining 24 cells (5 arms × 5 seeds minus this
one) **have not been run** — see deviation note below for why and what
this implies for **P2-go/no-go**.

### Exact commit + config

* Code commit: `c02be92` on `lpx/wodd`
  (`scripts/phase2/launch_arm.sh` + `examples/phase2_sft.py` unified
  entrypoint, registered against the WoDD trainer for `arm=wodd`).
* Arm: **A4 WoDD** (`llada_wodd`, `model_type=llada_wodd`).
* Backbone: `GSAI-ML/LLaDA-8B-Base` (frozen-tokenizer; weights warm-
  start, all params trainable per WODD_PLAN §3.2).
* Partition: `prelude_layers=8, recurrent_layers=16, coda_layers=8`.
* Tape: `tape_dim=1024, tape_max_writes=16, tape_n_heads=8`.
* Train-time `T_step` schedule: uniform over `[1, 4]` (eval default
  `t_step_eval=4`); `zero_init_wodd=True` so the WoDD modules are
  exact-identity at init (validated in Phase 1a, max logit diff =
  `0.0`).
* Phase 1b gate-collapse mitigation enabled: `gate_entropy_lambda=1e-2`,
  `write_sparsity_lambda=1e-2` (per the `EXTRA_WODD_FLAGS` block in
  `scripts/phase2/launch_arm.sh`).
* Data: `.data/sft/llada/openmath2-500k` (pre-tokenized OpenMath
  derivative), `200K` examples, **1 epoch** (deviation from script
  default — see below).
* Optimizer: AdamW, `lr=2e-5`, `warmup_ratio=0.0`, `weight_decay=0.0`
  (script defaults), `bf16=True`, FSDP via
  `scripts/accelerate_configs/fsdp.yaml`, `gradient_checkpointing=True`,
  `per_device_train_batch_size=2`, `gradient_accumulation_steps=1`,
  `8×H800` → global batch 16, total steps 12492.
* Save: `save_steps=2000, save_total_limit=2`; `checkpoint-final`
  sharded into 4 safetensors (FSDP all-gather on save).

### Raw training metrics (no rounding)

End-of-run dict from the HF Trainer (final progress line, log file
`.models/phase2/wodd/seed0/phase2_pilot.log`):

```
{'train_runtime': 7206.3801,
 'train_samples_per_second': 27.734,
 'train_steps_per_second': 1.733,
 'train_loss': 0.5079153526146137,
 'epoch': 1.0}
```

Final step-12450 log dict (last `logging_steps=50` log emitted before
end-of-run, T_step sample=2 at that step):

```
{'wodd/T_step': 2.0,
 'wodd/mean_gate': 0.3017578125,
 'wodd/gate_entropy': 0.6117557287216187,
 'wodd/loss_mdlm': 0.10263676196336746,
 'wodd/loss_sparsity': 0.3017578125,
 'wodd/loss_entropy': 0.6117557287216187,
 'wodd/tape_contrib_rel_norm_iter0': 0.7659505605697632,
 'epoch': 1.0}
```

Gate dynamics (gate-collapse health, the Phase 1b kill criterion):

* `mean_gate ≈ 0.30` averaged over the last 200 logging windows — well
  above the Phase 1b collapse threshold (`< 0.05`); the mitigation
  worked at full-scale FSDP, as predicted by the 100-step harness in
  Phase 1b.
* `gate_entropy ≈ 0.61` (max `≈ 0.693`); the gate distribution stays
  diffuse, not bimodal-saturated.
* `tape_contrib_rel_norm_iter0 ≈ 0.74` — the tape-read residual is a
  real signal at the read site (not zero, not blowing up the block
  output norm).

### Eval (in-flight)

Launched `bash scripts/phase2/eval_arm.sh wodd
.models/phase2/wodd/seed0/checkpoint-final 8 0` with
`HF_HUB_OFFLINE=0 https_proxy=http://localhost:20172` to fetch the
eval sets through the cluster's working proxy (port 20172 — port
20173 intercepts and SSL-fails). Tasks in order: `gsm8k_cot` (5-shot,
greedy, `cfg_scale=0.0`, `max_new_tokens=512`), `minerva_math`
(4-shot, MATH-500 proxy), `humaneval_instruct_llada` (0-shot),
`mbpp_instruct_llada` (3-shot). Observed ~`60s/step` at
`per_device_batch_size=1`, ~165 batches/rank × 8 ranks for GSM8K
→ ~2:45h for GSM8K alone, ~10–12h for all four tasks. Numbers will be
appended in a follow-up commit.

### Deviation from the plan, with reason

The §3.2 grid spec is **5 arms × 5 seeds = 25 cells**. The above is
**cell 1 of 25**. The other 24 cells have **not** been run because the
realized per-cell cost is ~2 h training + ~10–12 h eval ≈ ~14 GPU-h on
the available 8×H800, giving a full-grid cost of ~350 GPU-h — a single
interactive session cannot deliver that. Two further deviations:

1. **1 epoch instead of script default 3.** The launch override
   reduced `num_train_epochs` from `3` to `1` to fit the pilot inside
   the night-time budget. This is the per-cell cost driver; restoring
   3 epochs triples the per-cell time. All published numbers from
   this cell will be at 1 epoch — they will **not** be directly
   comparable to a future cell run at 3 epochs.
2. **OpenMath subset = 200K (script default would have been 500K).**
   Same reason. The 200K subset was selected with the same recipe
   used in the Phase 1b harness and is consistent with the data
   builder under `.data/sft/llada/openmath2-500k[train:500000,test:5000]`,
   sliced down at the trainer.

### Interpretation, with prediction tagging

* **P2-go/no-go (WoDD ≥ MetaState + 3 pp on GSM8K)** is **not yet
  testable**: the MetaState arm has not been trained. With WoDD as
  the only cell, the headline pre-registered prediction of Phase 2
  remains **OPEN**.
* What this cell **does** confirm at GPU scale, independent of the
  comparison: (i) the zero-init invariant holds end-to-end through
  FSDP/bf16 (training is stable from step 1 with no warm-up of the
  WoDD heads), (ii) the gate-entropy mitigation from Phase 1b
  transfers to full-rank training (no collapse), and (iii) the tape
  carries non-trivial signal (`tape_contrib_rel_norm > 0.7`). None of
  these are P2 predictions, but they are precondition checks that
  had to hold for the P2 comparison to be meaningful — they hold.

### What it would take to close P2

* **Minimum publishable comparison**: train one MetaState seed=0 cell
  at the matched config (~2 h training) + run GSM8K only on both
  checkpoints (~3 h each) = ~8 GPU-h additional. That gives a single-
  seed WoDD-vs-MetaState ΔGSM8K. **Cannot bound P2 noise**, but the
  plan still allows reporting it as a single-seed point estimate.
* **Full §3.2 comparison**: 5 arms × 5 seeds × 1 epoch × (train+eval)
  = ~350 GPU-h ≈ 2 days of continuous 8×H800. Out of scope for one
  session.

### Phases 3–6: not started, blocked on Phase 2

All four downstream phases (capacity sweep, linear probe, Dream-7B
replication, falsification battery) depend on having at least one
matched WoDD-vs-MetaState comparison from Phase 2; running them on
the WoDD-only cell would not test the pre-registered predictions.
Their **infrastructure** is in place and tested at small scale (see
"Phase 2-6 launcher + eval infrastructure" section above) but no
training/eval has been executed for them in this session.

### Phase 2 numbers — WoDD seed=0 GSM8K (2026-05-14)

**GSM8K (gsm8k_cot, 5-shot, greedy, cfg_scale=0.0, max_new_tokens=512,
n=1319 full test set)**:

| metric          | exact_match       | stderr            |
|-----------------|-------------------|-------------------|
| flexible-extract| 0.13874147081122062 | 0.009521649920798143 |
| strict-match    | 0.009097801364670205 | 0.002615326510775672 |

Eval wall time: `10174.4s ≈ 2:50h` on 8×H800 at
`per_device_batch_size=1, max_new_tokens=512, steps=512, block_size=512`.
Raw JSON: `.eval/phase2/wodd/seed0/gsm8k_cot_2026-05-14T06-01-06.json`.

**MATH-500 (minerva_math)**: **skipped** — `lm-eval[math]` extras are
not installed in the cluster env (the task requires
`antlr4-python3-runtime==4.11`, `sympy`, `math_verify`). The user-stated
constraint "环境已经配好了，不要修改环境了" forbids `pip install`,
so MATH-500 is unavailable via the harness in this env. Re-running with
`pip install -e lm-evaluation-harness[math]` would unblock it on a
future session.

**HumanEval / MBPP**: **not run** in the same eval batch — `set -e` in
`scripts/phase2/eval_arm.sh` propagated the minerva_math failure and
killed the script before humaneval/mbpp tasks ran. The script invokes
them after minerva. They can be re-run independently against the same
checkpoint, but with the comparison-first strategy (see below) they
are deferred until at least one MetaState GSM8K number lands so the
P2-go/no-go gap is testable.

### Comparison-first strategy (deviation, with reason)

Per WODD_PLAN.md the P2 headline prediction is GSM8K-specific:
"P2-go/no-go: WoDD ≥ MetaState + 3 pp on GSM8K." With the realized
per-cell cost of ~14 GPU-h, completing all 4 tasks for WoDD-only
would consume ~9 more GPU-h without producing any cross-arm test;
running MetaState seed=0 (~2 h train + ~3 h GSM8K eval = ~5 h)
instead **directly tests the headline prediction**.

Decision taken: launch MetaState seed=0 training **now** on the same
config as the WoDD pilot (`200K OpenMath × 1 epoch, lr=2e-5,
warmup_ratio=0.0, per_device_bsz=2, grad_accum=1, bf16+FSDP,
gradient_checkpointing=True`), then GSM8K eval at the same decoding
config. Result lands in a follow-up commit.
