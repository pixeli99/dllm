# Looped LLaDA Idea Loop Record

Date: 2026-05-08

Local repo root: `/Users/pixeli/dllm`

Cluster repo root: `/lustre/projects/polyullm/lipengxiang_tmp/dllm`

## Direct Answer

I do not have 100% confidence in the current loop design as a paper-worthy
idea yet.

The current design is technically coherent and promising, but it is still an
unproven hypothesis. The strongest version of the idea is not "add a loop";
it is:

> A pretrained diffusion LM can be retrofitted with a learned latent feedback
> transition so that extra inference-time recurrent compute improves denoising
> and reasoning without adding a second model or changing the tokenizer/data
> interface.

That becomes paper-worthy only if the experiments show a clean accuracy vs
compute story, robust T_rec generalization, and ablations that rule out "just
more SFT" or "just more trainable parameters."

## Current Code Anchors

- Main SFT entry: `/Users/pixeli/dllm/examples/llada_looped/sft.py`
- Looped trainer: `/Users/pixeli/dllm/dllm/core/trainers/mdlm_looped.py`
- Looped model: `/Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py`
- Looped config: `/Users/pixeli/dllm/dllm/pipelines/llada_looped/models/configuration_llada_looped.py`
- Main run script: `/Users/pixeli/dllm/scripts/looped_llada/run.sh`
- R-unfrozen run script: `/Users/pixeli/dllm/scripts/looped_llada/run_unfrozen_r.sh`
- Two-stage scripts:
  `/Users/pixeli/dllm/scripts/looped_llada/run_stage1_frozen.sh` and
  `/Users/pixeli/dllm/scripts/looped_llada/run_stage2_unfrozen.sh`
- GSM8K eval script: `/Users/pixeli/dllm/scripts/looped_llada/eval_gsm8k.sh`

## Current Loop Design

The implemented V2.1-a loop is:

```text
h_0 = prelude(x)
h_1 = R(h_0)
h_r = R(RecursiveLink(h_{r-1}, r-2)) for r = 2..T_rec
logits = head(coda(h_T_rec))
```

Important properties:

- `T_rec=1` is intended to be exactly vanilla split LLaDA.
- `RecursiveLink` is bypassed on the first pass and only applies to feedback
  transitions.
- The model uses full BPTT through all recurrent sweeps.
- The default trainable surface freezes the backbone and trains only
  `recursive_link.*`.
- The loss is computed only on final logits, not on intermediate loop states.
- `T_rec` is sampled per batch during training.

## Confidence Assessment

What looks strong:

- The first-pass bypass gives a clean identity baseline at `T_rec=1`.
- The residual `RecursiveLink` is a minimal intervention and gives a plausible
  attribution story.
- Iter-conditioned FiLM directly addresses the observed "T=2 trap" where one
  transition operator may not fit both early reshaping and later refinement.
- The evaluation script can sweep `mu_rec_eval`, so the idea naturally supports
  an anytime-compute curve.

What blocks 100% confidence:

- Frozen `R` may not be an iterable refinement operator. A pretrained middle
  stack was not trained to consume its own outputs repeatedly.
- Final-only loss may teach the last state to work without forcing each loop
  step to become a meaningful refinement.
- Training over `T_rec in [2, 6]` does not guarantee inference-time monotonicity
  or extrapolation to larger `T_rec`.
- The current headline frozen-link setup has a smaller trainable surface than a
  full SFT baseline, so comparisons must be carefully matched.
- I found and corrected stale comments that described the loop as truncated
  BPTT even though the implementation uses full BPTT.
- `/Users/pixeli/dllm/scripts/looped_llada/run_stage1.sh` used to launch a
  `T_rec=1` warmup that could not train `RecursiveLink`; it is now explicitly
  deprecated so it fails before wasting GPU time.
- There are no local experiment artifacts under `/Users/pixeli/dllm/.models`
  available in this checkout, so this assessment is code-based rather than
  result-based.

Local verification note:

- Syntax check passed for the looped trainer/model/config/SFT files with local
  Python.
- A torch forward/backward sanity check could not run locally because the
  expected conda env `/Users/pixeli/miniconda3/envs/dllm` is absent.

## Idea Loop

### Iteration 0: Current V2.1-a Feedback Bridge

Hypothesis:

`RecursiveLink` can transform each recurrent output into a better input for the
next shared `R` pass, making extra compute useful at inference time.

Keep this as the baseline idea, but do not stop here. It is paper-worthy only if:

- `T_rec=2..6` beats `T_rec=1` consistently.
- The gain remains after comparing against full-param SFT and surface-matched
  ablations.
- Accuracy improves faster than cost increases on at least one reasoning-heavy
  benchmark.

### Iteration 1: Train R To Be Iterable

Hypothesis:

The loop will only work if the middle stack `R` is trained as a recurrent
operator, not merely reused frozen.

Use:

- `/Users/pixeli/dllm/scripts/looped_llada/run_unfrozen_r.sh`
- `/Users/pixeli/dllm/scripts/looped_llada/run_stage1_frozen.sh`
- `/Users/pixeli/dllm/scripts/looped_llada/run_stage2_unfrozen.sh`

Paper value:

If this wins clearly over frozen `R`, the story becomes "fine-tuning teaches
the pretrained middle layers to become an iterative denoising/refinement
operator."

Risk:

If this only matches full-param SFT, the loop is not independently valuable.

### Iteration 2: Add Intermediate Supervision

Hypothesis:

Final-only loss is too weak. Each loop state should be a valid denoising state,
or at least should improve relative to the previous state.

Candidate code direction:

- Add trainer/model support for optional intermediate loop logits.
- Add `loop_aux_loss_weight`.
- Add modes:
  - `final`: current behavior.
  - `all`: loss on every loop state.
  - `sampled`: loss on one random intermediate state plus final state.

Paper value:

This would let the paper argue for recurrent denoising as a learned process,
not just a deeper unrolled network.

Main risk:

Running coda/head at every loop state increases memory and compute. The first
implementation should probably support `sampled` before `all`.

### Iteration 3: Adaptive Compute / Anytime Decoding

Hypothesis:

Different examples and diffusion stages need different recurrence depth.

Candidate signals:

- `loop/residual_norm_iter*`
- `loop/adapter_update_norm_iter*`
- mask-token confidence during sampler steps
- entropy over masked positions

Paper value:

A static `T_rec` sweep is useful, but adaptive compute is a stronger paper:
the model spends extra recurrence only where uncertainty or residual movement
is high.

### Iteration 4: Couple Recurrence With Diffusion Time

Hypothesis:

The best recurrence depth depends on diffusion timestep/noise level. Heavily
masked inputs may need more latent sweeps; near-clean states may need fewer.

Candidate directions:

- Condition `RecursiveLink` on both feedback index and diffusion timestep `t`.
- Sample `T_rec` as a function of mask ratio.
- During sampling, use more `T_rec` in early block steps and less in late steps.

Paper value:

This connects recurrence to the diffusion process itself, which is more specific
than generic weight sharing.

### Iteration 5: Partition Search

Hypothesis:

The current `8/16/8` prelude/R/coda split may be arbitrary. Different splits
change the semantic level of the recurrent state.

Minimal grid:

- `4/24/4`: more compute in shared recurrent operator.
- `8/16/8`: current default.
- `12/8/12`: higher-level short loop.
- `0/24/8` or `8/24/0` only if the architecture supports the boundary safely.

Paper value:

Partition sensitivity can reveal whether recurrent refinement should happen in
lower, middle, or high-level layers.

## Rejected Or Low-Priority Ideas

- `T_rec=1` link warmup is not valid under first-pass bypass because
  `RecursiveLink` is not called.
- Pure frozen-`R` + link-only training should not be the final claim unless it
  produces surprisingly strong results; it is best treated as an attribution
  baseline.
- A single GSM8K result is not enough. The paper needs a compute-quality curve
  and at least one additional reasoning or generation benchmark.

## Experimental Loop To Repeat

1. Run a cheap sanity pass.

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /lustre/projects/polyullm/lipengxiang_tmp/dllm
pip install -e /lustre/projects/polyullm/lipengxiang_tmp/dllm
python -m py_compile \
  /lustre/projects/polyullm/lipengxiang_tmp/dllm/dllm/core/trainers/mdlm_looped.py \
  /lustre/projects/polyullm/lipengxiang_tmp/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py \
  /lustre/projects/polyullm/lipengxiang_tmp/dllm/examples/llada_looped/sft.py
```

2. Run one training candidate.

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /lustre/projects/polyullm/lipengxiang_tmp/dllm
bash /lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/run_stage1_frozen.sh
bash /lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/run_stage2_unfrozen.sh
```

3. Sweep inference recurrence depth.

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /lustre/projects/polyullm/lipengxiang_tmp/dllm
ckpt=/lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-v2.1a-two-stage/checkpoint-final
for t in 1 2 3 4 5 6 8; do
  bash /lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/eval_gsm8k.sh \
    --checkpoint "${ckpt}" \
    --t_rec "${t}" \
    --num_gpu 4 \
    --batch_size 1 \
    --max_new_tokens 256 \
    --steps 256 \
    --block_size 64
done
```

4. Compare against required baselines.

Required baselines:

- Vanilla full-param SFT:
  `/Users/pixeli/dllm/scripts/looped_llada/run_baseline.sh`
- Surface-matched V1 pure recurrence:
  `/Users/pixeli/dllm/scripts/looped_llada/run_v1.sh`
- Frozen `R` V2.1-a:
  `/Users/pixeli/dllm/scripts/looped_llada/run.sh`
- R-unfrozen V2.1-a:
  `/Users/pixeli/dllm/scripts/looped_llada/run_unfrozen_r.sh`
- Two-stage frozen-link then R-unfrozen:
  `/Users/pixeli/dllm/scripts/looped_llada/run_stage1_frozen.sh` and
  `/Users/pixeli/dllm/scripts/looped_llada/run_stage2_unfrozen.sh`

5. Decide.

Continue the loop only if at least one candidate satisfies:

- `T_rec>1` beats `T_rec=1` by a meaningful margin.
- Gains survive a fair trainable-surface comparison.
- The accuracy-vs-compute curve has a useful operating point.
- Loop diagnostics do not show collapse, explosion, or a single-step-only trap.

Stop or pivot if:

- Higher `T_rec` consistently hurts.
- Unfreezing `R` only reproduces full-param SFT with more compute.
- Gains disappear when sampling/eval settings are held fixed.

## Paper-Worthy Shape

A valuable paper should be framed around:

> Learned latent recurrence for diffusion language models: retrofitting a
> pretrained DLM with an iterative hidden-state refinement operator that trades
> inference compute for reasoning quality.

Minimum evidence:

- Accuracy vs `T_rec` sweep on GSM8K and at least one harder math/reasoning set.
- Accuracy vs FLOPs or wall-clock curve.
- Baseline comparison to vanilla SFT with the same data.
- Surface-matched ablations.
- FiLM vs no-FiLM ablation.
- Frozen `R` vs unfrozen `R` ablation.
- Final-only vs intermediate-supervision ablation if implemented.
- Qualitative loop diagnostics showing iterative refinement rather than drift.

Current recommendation:

Do not bet the paper on the frozen-link-only V2.1-a result. Use it as a clean
baseline, but the best next bet is the two-stage R-unfrozen path plus an
intermediate-supervision variant if the first T_rec sweep is not clearly
monotonic.
