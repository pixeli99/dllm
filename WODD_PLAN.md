# WoDD: Experimental Plan and Paper Storyline

**Code branch**: `lpx/wodd` (this branch).
**Status (2026-05-13)**: pipeline + trainer + 21 passing unit / invariant tests committed; ready for Phase 1 GPU runs.

---

## 0. The frame: what story this paper is actually telling

The cliché version of this paper would be "we added latent memory to a
diffusion LM, accuracy on GSM8K went up". That story has been told four
times already in the last 12 months (MetaState, LATTS, DPad, LaDiR), and
the lifts are all in a narrow band around 3–5 pp. Telling it a fifth time
adds nothing.

The actual story we want to tell is:

> **There is a structural reason latent reasoning in diffusion LMs caps
> at ~5 % gain, and it isn't capacity, training signal, or scale. It is
> the type of memory.** All four existing methods do *state evolution* —
> a fixed-capacity buffer that the model overwrites every step. Token
> CoT scales orders of magnitude better because it is *append-only*:
> each step writes new bits and never erases. We argue that this
> distinction is fundamental, not stylistic; demonstrate it experimentally
> by isolating the append-only-versus-evolution variable while holding
> everything else fixed; and show that a tiny architectural change
> (WoDD) restoring append-only semantics produces a predictable scaling
> law that state-evolution methods cannot reach regardless of how much
> capacity or compute they are given.

If we can land all three pieces of that claim — *type-of-memory matters*,
*it explains the plateau*, *and WoDD is the minimal way to fix it* — the
paper is a science contribution about a generative-model substructure,
not just a leaderboard entry. If we land only the first two pieces and
WoDD turns out to be subsumed by an even simpler trick, that is still a
publishable negative result with positive value to the field — and we
should be ready to write it that way.

Two design decisions follow from this frame:

1. **The hero figure is a *scaling law*, not a bar chart.** We will pre-
   register the predicted scaling curve in §4 and tell the paper as
   confirmation/refutation of that prediction. A bar chart of "WoDD beats
   MetaState by 3 pp on GSM8K" is the *secondary* result; the primary
   result is the predicted relationship between effective bit budget and
   reasoning accuracy.

2. **Negative results are non-decorative.** Section 6 will publish at
   least three settings where WoDD does **not** help (short-context
   tasks, perception-like generation, tasks below the bit-budget
   threshold). This is what distinguishes the paper from "another DLM
   memory method"; we are claiming we *know when* the mechanism applies.

---

## 1. The intellectual setup the paper has to defend

### 1.1 Two opposing programs

There is a real tension in late-2025 generative modeling. We don't have
to invent it; it is on the literature's face:

* **Iteration-collapsing program** — MeanFlow (Geng et al., NeurIPS 2025
  Oral), Drifting (Deng et al., 2602.04770), ELF (Hu/Qiu/Andreas/He,
  2605.10938). For images, 1-NFE generation now matches multi-step
  diffusion. Lesson: inference iteration is a crutch when the data
  manifold is smooth.

* **Iteration-growing program** — Saunshi (ICLR 2025), Huginn (Geiping,
  NeurIPS 2025), Ouro (33-author, 2025-10), HRM (Sapient + Tsinghua),
  MoR (NeurIPS 2025). For reasoning, more inference iterations beat
  more parameters on the matched-FLOPs Pareto.

These programs make incompatible empirical predictions on the same
substrate: language. The paper's first move is to argue that they are
not actually in conflict — they describe different *modes* of inference
work — and that **the structural feature distinguishing the two modes
is whether information accumulates append-only or not**. Images don't
need append-only thinking because perception isn't a build-up of facts.
Math does, because the value of "x+y" depends on writing "x" and "y" down
first. WoDD is built to test exactly this distinction.

This framing also organizes related-work cleanly. State-evolution
methods (MetaState, LATTS, Coconut, DPad, LaDiR, ReMDM) sit on the
iteration-growing side of the axis, but with the wrong memory type —
they iterate but don't accumulate. WoDD shares iteration with them but
flips the memory type. That is the precise novelty.

### 1.2 Why the ~5 % plateau is information-theoretic

Token CoT writes log|V| ≈ 17 bits per step into the context. After T
steps the total information addressable for downstream prediction is
T · log|V|, which is unbounded in T.

State evolution methods (MetaState's GRU on a 64×1024 buffer is the
cleanest case) update the buffer in place. The capacity is dim(state) ·
bits-per-coord ≈ 65 K floats. GRU updates are contractive in expectation
(the gates pull the buffer toward an attractor), so the *effective*
information surviving K updates is sub-linear in K — formally bounded
above by the buffer dimension regardless of how many iterations you
run. This is the capacity ceiling we observe empirically.

Append-only writes — Memorize-then-attend — preserve every write,
contribute additive log|V|-equivalent bits per write, and are
information-theoretically equivalent to token CoT modulo embedding
geometry. They escape the contraction bound.

§3 of the paper makes this argument formal with a small theorem
following Khrulkov-style information-bottleneck analysis. It is one
page, not the centerpiece — but it explains both why the plateau exists
and why our intervention is the *minimal* fix.

### 1.3 The minimum intervention

A reviewer asks: "Couldn't you achieve the same with bigger MetaState?"
A bigger MetaState raises the buffer ceiling but does not change the
contraction property — it merely delays the plateau. The plateau is
qualitatively in the same place, just at a higher dim. Our scaling-law
experiment in §4 is built specifically to distinguish these.

A reviewer asks: "Couldn't you just use more denoising steps?" More
denoising steps revealed more tokens — they ARE append-only writes, in
output space. But they are writes the user observes; they are
constrained by the vocabulary, by token-position grids, by the
unmasking-order schedule. WoDD provides append-only writes that are
unconstrained by any of these — in particular, they are continuous and
they are not part of the output. This is what makes them function as
*thought*.

WoDD is, by design, the smallest possible architectural addition that
restores append-only semantics: one write head, one tape cross-attention
module, one append-only list. If a simpler fix exists, we want to find
it; this is why §6 includes deliberate attempts to falsify WoDD.

---

## 2. The mechanism (code already implements this)

See `examples/llada_wodd/README.md` for the operational description. In
the paper we will introduce only one new symbol:

  Definition (Append-only latent CoT). For a transformer-based DLM with
  T_step inner iterations per denoising step, an *append-only latent
  trace* is a sequence M = (v_1, ..., v_{T_step}) of vectors in R^d such
  that for all k < k', M_k is computed before M_{k'} and M_k is
  immutable after creation. The model may read M_{1..k-1} when
  computing v_k but cannot modify M_{1..k-1}.

WoDD implements this with a per-step write gate g_k ∈ [0, 1]; the
appended entry is g_k · v_k. We will show in §3 that this is equivalent
to an append-only trace with a soft inclusion mask, not a relaxation of
the append-only property itself.

---

## 3. Experimental plan: 8 weeks, six phases

Compute envelope: assumed 8×H100 for phases 1–3, 16×H100 for phase 4–5,
1×A100 sufficient for phase 6 (interpretability) and §7 (probing).
Numbers below are for one configuration; matched-compute baselines and
seed replication are listed under each phase.

### Phase 1 — Implementation verification (week 1)

**Goal**: confirm the WoDD code reproduces vanilla split LLaDA at step 0.
**Status**: unit + invariant tests pass in this branch (21/21). One more
GPU-level check needed.

* Run 1a: `T_step=1, zero_init_wodd=True` on `LLaDA-8B-Base` → MDLM eval
  loss on a held-out OpenMath subset should be **bit-identical** to the
  base LLaDA. Tolerance: max absolute logit diff < 1e-4 in fp32.

* Run 1b: 100 SGD steps of the WoDD trainer with `T_step=2` on a tiny
  subset (e.g., 1 K samples). Assert `wodd/mean_gate` is non-trivial
  (> 0.05) and `wodd/gate_entropy > 0.1` — i.e., the gate didn't collapse.

**Kill criterion**: if Run 1a fails (logits diverge at init), the
zero-init invariant is broken, training is unsafe. Fix before Phase 2.

### Phase 2 — Matched-FLOPs head-to-head (weeks 2–4)

**Goal**: the headline number. WoDD vs the three nearest neighbors on
math reasoning, at matched FLOPs.

Backbone: frozen `LLaDA-8B-Base`. Trainable surface for each method:
the bolt-on module only. Same OpenMath subset (500 K train / 5 K eval),
same optimizer (AdamW), same LR schedule, same training tokens.

| Arm | Adapter | Trainable params | Inner step count | Notes |
|---|---|---|---|---|
| A0 | none (vanilla LLaDA) | 0 | 1 | reference |
| A1 | LATTS | 0 | 4 | latent test-time steps only |
| A2 | MetaState | ~0.6 % | 4 | persistent fixed memory |
| A3 | DPad | ~1 % | 1 | suffix scratch |
| **A4** | **WoDD** | ~1 % | 4 | append-only tape |

For each arm we report on GSM8K and MATH-500 (in-distribution math),
HumanEval and MBPP (out-of-distribution code). Sample-efficient
evaluation: 5 random seeds per cell, pass@1 with greedy decoding plus
self-consistency at N=16 for variance estimation.

**Pre-registered prediction**: WoDD ≥ MetaState + 3 pp on GSM8K under
matched FLOPs. If the gap is < 1 pp, the *append-only-vs-state-
evolution* hypothesis is falsified at this scale.

We will report whichever direction the result goes — including a clean
write-up if WoDD does not beat MetaState. A null result here would be a
strong negative finding about the *type-of-memory* claim and is
publishable as such.

### Phase 3 — The scaling-law experiment (weeks 4–6, parallel with Phase 2)

This is the section we expect reviewers to remember.

**Setup**: fix everything except *effective bit capacity*. For both
MetaState and WoDD, sweep the relevant capacity knob:

* MetaState: memory slots M × dim D ∈ {64×1024, 128×1024, 256×1024,
  512×1024, 1024×1024} (5 points).
* WoDD: tape_dim D × T_step T ∈ {(256, 4), (512, 4), (1024, 4), (1024,
  8), (1024, 16)} (5 points).

We hold the *trainable parameter count* roughly constant within each
method (so the comparison isolates capacity, not scale). On the x-axis
we plot the *effective addressable bit budget* — for MetaState, M · D
(capped by GRU contraction); for WoDD, T · D.

**Pre-registered shape**:

* MetaState curve: rises monotonically and **saturates well before the
  highest capacity point**. Specifically, the slope between 64 and 256
  slots should be ≥ 4× the slope between 256 and 1024 slots.
* WoDD curve: rises monotonically and **does not saturate** within the
  swept range, until compute (not capacity) becomes the bottleneck.

If MetaState's curve does not saturate, or WoDD's does, our framing is
wrong. We report both curves in the paper figure regardless.

This is the figure we tell the story around. The bar chart from Phase 2
is the lower-right panel.

### Phase 4 — Probing the tape (week 6)

**Goal**: a science result independent of accuracy. If WoDD's tape is a
working scratchpad rather than a noise sink, its contents should be
*meaningful*.

Setup: take a converged WoDD-8B (best from Phase 3). For each problem
in a held-out GSM8K subset, run the model and record the tape entries
(v_1, ..., v_{T_step}). Run an AR teacher (Qwen3-8B-Math) on the same
problems and record the CoT trace.

Train a small linear probe to predict, from v_k, which AR-CoT token
class is "active" at that point. (Token classes: digits / operators /
variable-binding words / answer-form tokens.)

**Predictions**:
* Probe accuracy > 50 % beats chance significantly (random over 5
  classes ~ 20 %).
* Probe accuracy increases with k (later writes are more
  semantically loaded).
* For *hard* problems (long AR-CoT trace), tape entries cluster
  closer to high-arithmetic-density CoT tokens.

If the probe is at chance, WoDD's tape is decorative even when it
boosts accuracy — that is still useful information for the field; we
publish it.

### Phase 5 — Cross-DLM transfer (week 6–7)

Replicate the Phase 2 head-to-head on Dream-7B (different DLM family).
If WoDD only works on LLaDA, the conclusion is much narrower; we have
to flag this. Two random seeds on a smaller eval (GSM8K + MATH-500)
sufficient.

### Phase 6 — Deliberate falsification (week 7)

Three pre-registered settings where WoDD should **not** help, by
hypothesis. Publishing them strengthens the type-of-memory argument:

* **Short-output fluency** (e.g., GLUE-style classification). Tape
  writes have no time to amortize. Predicted: WoDD ≈ baseline.
* **Single-step retrieval** (TriviaQA, closed-book QA short answers).
  No iterative information accumulation needed. Predicted: WoDD ≈
  baseline.
* **Sub-threshold reasoning** (1-step math, 2-digit add). Required bit
  budget < D. WoDD's curve flat below this threshold. Predicted: WoDD
  ≈ MetaState ≈ baseline below the threshold, separates above.

A particularly satisfying paper has the bit-budget threshold appear
in figure 4 as a *predicted critical value*; if the empirical critical
value is within a factor of 2 of the prediction, that is a precise
science result.

### Phase 7 — Writing (week 8)

The paper outline below. Targeting NeurIPS 2026 cycle.

---

## 4. Paper outline (deliberately non-cliché)

The standard six-section template (intro / related / method / results /
ablation / conclusion) buries the actual claim. Below is the structure
we will use:

1. **The plateau** (≤ 2 pages)
   Empirical observation: four 2025–2026 papers report ~5 % math-
   reasoning lift from latent memory in DLMs, despite very different
   designs. We argue this is not a coincidence — it is a structural
   ceiling.

2. **Information accumulates, or it doesn't** (≤ 2 pages)
   The append-only-vs-evolution distinction made formal. One small
   theorem (capacity bound on contractive memory). Token CoT placed on
   the append-only side. The two-program tension (Kaiming pole vs Loop
   pole) recast as a question of which mode of inference the *task*
   requires.

3. **WoDD** (≤ 2 pages)
   Method described as the minimum intervention restoring append-only
   semantics. Architecture diagram + one equation per module + the
   identity-at-init guarantee. The whole section can be ~1.5 pages
   because the mechanism is intentionally small.

4. **The scaling law** (≤ 3 pages)
   Phase 3 result is the headline figure. Saturation vs non-saturation
   visible at a glance. Bit-budget threshold annotated. Phase 2 numbers
   appear as one panel; Phase 5 cross-DLM as another.

5. **What is in the tape** (≤ 2 pages)
   Phase 4. Linear probe accuracy curves. Three case studies (one per
   problem-difficulty band) showing tape entries clustering near
   AR-CoT semantic classes. This is the science section.

6. **When append-only doesn't help** (≤ 1.5 pages)
   Phase 6. Three falsification experiments. This is where most papers
   would skip; we give it almost as much space as the headline result.

7. **What WoDD does not solve** (≤ 0.5 pages)
   Length scaling beyond ~32 writes. RL fine-tuning interactions (likely
   broken, per Ouro experience). Multimodal extensions. Honest list.

8. **Implications** (≤ 1 page)
   The append-only-vs-evolution axis as a lens to re-read the latent-
   reasoning literature. Predictions for future work. Connection to the
   Kaiming-pole / Loop-pole framing.

We will deliberately **not** include:
* a "scaling to larger models" claim we can't actually support at our
  compute budget;
* a "WoDD + RL" section unless we can ship working code (Ouro's RL
  pipeline failure is the cautionary tale);
* a "we're SOTA" framing — the paper is not about being best on a
  benchmark, it's about explaining a structural phenomenon.

---

## 5. Risks, and how we will handle each

* **Risk A** — WoDD performs identically to MetaState at matched
  capacity, falsifying the type-of-memory claim. **Handling**: write the
  paper anyway, as a precise negative result. The capacity scaling-law
  experiment (Phase 3) still provides a unified explanation for the
  plateau and is publishable as "capacity, not type, is the bottleneck".

* **Risk B** — WoDD beats MetaState only with full backbone unfreezing,
  not with adapter-only training. **Handling**: report both, declare
  what the minimum intervention is honestly. This is what reviewers
  asked of Ouro; pre-empt it.

* **Risk C** — Gate collapses to always-off (model never writes) or
  always-on (model writes constantly). **Handling**: the entropy
  regularizer + gate-bias init is designed against this. If it still
  happens, anneal λ_entropy from high to low across training.

* **Risk D** — Tape probing returns chance accuracy. **Handling**:
  publish this finding. "Memory mechanism that helps reasoning without
  being interpretable" is itself a contribution (mirrors mechanistic
  interp results on AR LMs).

* **Risk E** — Vanilla LLaDA-8B is too weak a backbone for any latent
  mechanism to clearly help; the Phase 2 differences fall within
  seed noise. **Handling**: include Phase 5 (Dream-7B) as a cross-
  family check; if both backbones show null differences, the paper's
  claim narrows to "no current latent mechanism produces structural
  improvements at this scale" — which is also publishable and arguably
  more important than another marginal-positive paper.

The unifying response to all five risks: **we tell the story the data
supports, not the story we would have preferred**. The paper is
designed so that any of these outcomes still has a publishable shape.

---

## 6. Code state and dependencies

* `lpx/wodd` branch — pipeline, trainer, tests, SFT entry, run script.
* `21/21` unit + invariant tests pass in CPU sandbox; toy-model forward
  tests require `lm_eval` (present in real training env).
* GPU setup needed: standard dLLM FSDP config + ≥ 80 GB per GPU for
  T_step=4 with gradient checkpointing on LLaDA-8B.
* No new external deps; only adds files under `dllm/pipelines/llada_wodd/`
  and `dllm/core/trainers/mdlm_wodd.py`.

---

## 7. The "watch for" list during runs

What we want to see on TensorBoard (and what each signal means):

* `wodd/mean_gate` rising above its bias-init floor (~0.12) — the model
  is choosing to write.
* `wodd/gate_entropy` plateauing in [0.3, 0.6] — the gate isn't
  collapsing.
* `wodd/tape_contrib_rel_norm_iter{k}` rising monotonically — later
  iterations attend more heavily to earlier tape entries.
* `wodd/loss_mdlm` matching frozen-LLaDA baseline within the first 100
  steps — zero-init drop-in works at scale, not just in unit tests.

Failure signatures and what they mean:

* `mean_gate` stuck at 0.5 — gate not informative; check entropy reg
  coefficient.
* `mean_gate` near 0 or 1 — collapse; increase entropy reg.
* `loss_mdlm` immediately diverges from baseline — zero-init invariant
  broken in numerics (dtype issue, normalization issue).
* `tape_contrib_rel_norm` stays at 0 — tape is being written but cross-
  attn isn't reading; check `tape_read.out_proj` is unfreezing.

---

*Updated 2026-05-13. This document is the pre-registration of the
experimental program. Any deviation from the predictions stated in
sections 3 and 4 will be reported in the paper, regardless of whether it
makes the result look better or worse.*
