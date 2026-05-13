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

- Phase 2 actual training runs and eval (estimate: 150–250 GPU-h for the
  full 5×5 grid, plus eval).
- Phase 3 capacity-sweep launcher (small scaffold; reuses Phase 2 launcher
  with `--arm_kwargs_json '{"state_slots": M}'` / WoDD analogous).
- Phase 4 linear-probe pipeline (needs a converged WoDD-8B checkpoint
  from Phase 2/3 to extract `v_k`).
- Phase 5 Dream-7B port of the WoDD architecture.
- Phase 6 falsification tasks (GLUE / TriviaQA / 2-digit add).

