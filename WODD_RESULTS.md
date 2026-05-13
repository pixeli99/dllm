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

<!-- Phase 1 results below this line. -->
