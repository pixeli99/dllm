# WoDD Cluster Goal

Run the 6-phase WoDD experimental program on branch `lpx/wodd`. Full plan
in `WODD_PLAN.md` on the same branch.

## What to do

1. Check out `lpx/wodd`, install deps, verify `pytest tests/wodd/` passes
   (21 cases). Skipped forward tests should now run with `lm_eval`
   installed; report any new failures.
2. **Phase 1** — sanity. `T_step=1, zero_init_wodd=True` MDLM eval loss
   must match frozen LLaDA-8B-Base within fp32 1e-4. Kill if it doesn't.
3. **Phase 2** — matched-FLOPs head-to-head: A0 vanilla, A1 LATTS, A2
   MetaState, A3 DPad, A4 WoDD. Same OpenMath subset, same optimizer,
   same training tokens, 5 seeds. Report GSM8K, MATH-500, HumanEval, MBPP.
4. **Phase 3 (headline)** — capacity sweep:
   * MetaState slots ∈ {64, 128, 256, 512, 1024} × dim 1024.
   * WoDD `(tape_dim, T_step)` ∈ {(256,4),(512,4),(1024,4),(1024,8),(1024,16)}.
   Plot accuracy vs effective bit budget for both curves.
5. **Phase 4** — train a 5-way linear probe from WoDD tape entries `v_k`
   to AR-CoT token-class labels from a Qwen3-8B-Math teacher.
6. **Phase 5** — repeat Phase 2 on Dream-7B (GSM8K + MATH-500 only).
7. **Phase 6** — falsification: WoDD vs MetaState vs vanilla on (a) GLUE
   classification, (b) TriviaQA closed-book, (c) 2-digit addition.

## Pre-registered predictions (these are the kill criteria, not goals)

* **P2-go/no-go**: WoDD ≥ MetaState + 3 pp on GSM8K. Gap < 1 pp falsifies
  the type-of-memory hypothesis — report it cleanly anyway.
* **P3-shape**: MetaState saturates before 1024 slots; WoDD does not
  saturate within the swept range. Either failure breaks the framing.
* **P4-probe**: linear probe > 50 % on 5-way class (chance 20 %).
* **P6-falsification**: WoDD ≈ baseline on (a) and (b); WoDD ≈ MetaState
  below the bit-budget threshold on (c), separates above.

## Report format

For each phase, append to a file `WODD_RESULTS.md` on a sub-branch
`lpx/wodd-results`:
- exact commit + config used
- raw numbers (mean ± std over seeds), no rounding
- 1-paragraph interpretation tagging which prediction the result
  confirms / falsifies
- any deviation from the plan, with the reason

Push intermediate results after each phase; do not wait for completion.
Null and negative results are first-class — write them up the same way.

## Hard constraints

* Do not invent new methods or scope creep. Stick to the 6 phases.
* Do not skip Phase 1 sanity. Drop-in invariant must hold at GPU scale.
* Do not tune toward beating MetaState; same hparam sweep budget for all
  arms in Phase 2.
* If anything in `WODD_PLAN.md` is ambiguous, ask before guessing.
