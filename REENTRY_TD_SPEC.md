# ReEntry-TD Implementation Spec

Date: 2026-05-10
Branch: `lpx/reentry-td`
Local repo root: `/Users/pixeli/dllm`
Cluster repo root: `/lustre/projects/polyullm/lipengxiang_tmp/dllm`

---

## 1. Goal

Train a sub-5M-parameter correction module on top of a frozen task-SFT
LLaDA so that **trajectory distillation + manifold re-entry** lets the
student match a 256-step teacher with fewer outer denoising steps. The
research bet is that for discrete diffusion on math reasoning, you can
trade outer denoising steps for inner recurrent micro-steps when the
inner transition keeps the hidden state on the model's natural
embedding manifold.

This branch builds on the V2.1 looped LLaDA infrastructure already in
this repo. It does **not** replace V2.1 -- it is a parallel research
line that reuses the layer split, the retrofit loader, and the freeze
profile, but adds:

- a teacher trajectory cache,
- a coarse-KL distillation loss with a tail-mass bucket,
- per-step coda decoding inside the inner loop,
- a manifold re-entry transition (decode -> soft-embed -> prelude -> R)
  with a tiny correction module instead of a free-form latent link.

---

## 2. Naming Lock

To avoid clashing with V2.1 terminology already used in
`/Users/pixeli/dllm/LOOPED_LLADA_RESEARCH.md` and the looped trainer.

| Concept | Term used in this branch | Notes |
|---|---|---|
| Per-batch number of inner R-passes inside one diffusion step | `T_rec` | Same as existing repo |
| Number of diffusion sampling steps | `diffusion_steps` (sampler `steps`) | Avoid the word "outer" -- in V2.1 docs "outer" means R-loop outer |
| Teacher diffusion-step granularity | `teacher_steps`, default 256 | |
| Student diffusion-step granularity | `student_steps`, default 64-128 | Decided by Phase 3 |
| Mapping ratio teacher -> student | `teacher_stride = teacher_steps / student_steps` | |
| The simple zero-init residual link used in P1 (no FiLM) | `SimpleResidualLink` | **Not** `RecursiveLink`. The existing `RecursiveLink` is the V2.1-c gated module and must keep its name. |
| The (h, u, du, step)-conditioned link used in P2 | `CorrectionLink` | New, no naming conflict |
| Manifold re-entry transition `decode -> soft_emb -> prelude -> R` | `manifold_reentry_step` | New |

---

## 3. Repo Layout (integrated, not standalone)

All new code lands inside the existing tree so the registry, eval
harness, and dataset utilities continue to "just work":

```
dllm/
├── core/
│   ├── trainers/
│   │   ├── mdlm.py                         (existing, base MDLM trainer)
│   │   ├── mdlm_looped.py                  (existing, V2.1)
│   │   ├── td_distill.py                   ★ NEW  Phase 2/4/7 trainer
│   │   └── td_losses.py                    ★ NEW  coarse KL + tail bucket
│   └── samplers/
│       ├── mdlm.py                         (existing)
│       └── mdlm_deterministic.py           ★ NEW  fp32 + position-eps tie-break
├── pipelines/
│   └── llada_looped/
│       ├── models/
│       │   ├── modeling_llada_looped.py    (existing V2.1-c)
│       │   ├── modeling_no_reentry_td.py   ★ NEW  Phase 4
│       │   └── modeling_reentry_td.py      ★ NEW  Phase 7
│       └── cache_io.py                     ★ NEW  trajectory cache I/O
└── tools/
    ├── build_teacher_cache.py              ★ NEW  Phase 1
    ├── compressibility_probe.py            ★ NEW  Phase 3
    └── transition_swap_eval.py             ★ NEW  Phase 5/6

examples/llada_looped/
├── sft.py                                  (existing)
└── sft_td.py                               ★ NEW  TD distillation entry

scripts/looped_llada/
├── ... (existing V2.1 scripts)
└── td_*.sh                                 ★ NEW  one per phase

tests/td/
├── test_sampler_deterministic.py           ★ NEW
├── test_coarse_kl.py                       ★ NEW
├── test_reentry_zero_init.py               ★ NEW
├── test_norm_cap.py                        ★ NEW
└── test_cache_roundtrip.py                 ★ NEW
```

**Why integrate vs standalone**: cache_io and transition_swap eval are
useful tools for V2.1 too; eval harness auto-detects new model_type via
the `LLaDALoopedConfig` registry without per-phase wiring.

---

## 4. Phase 0: Cluster Prerequisites (mandatory pre-check)

Before any code change, verify on the cluster:

### 4.1 Teacher checkpoint
A vanilla full-param task-SFT LLaDA on OpenMathInstruct-2 must exist
and be loadable. Check:

```bash
ssh <cluster>
ls /lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-baseline*/checkpoint-final
```

If it does not exist, train one first via
`/lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/run_baseline.sh`
(this is **not** part of this spec; treat it as upstream work).
Acceptance: GSM8K pass at 256 diffusion steps reproduces vanilla
LLaDA-SFT-OpenMath2 numbers within 1%.

### 4.2 V2.1 checkpoints (optional, gates Phase 5)
Check for:
- `.models/loop_belief/openmath2-v2.1a/checkpoint-final` (frozen R)
- `.models/loop_belief/openmath2-v2.1a-two-stage/checkpoint-final` (R-unfrozen)

If present, Phase 5 smoke-swap runs in parallel. If absent, Phase 5 is
skipped and **only Phase 6 gates the decision** -- this is by design;
Phase 6 uses P1's own `h_r`, which is the primary representation we
care about.

### 4.3 OpenMathInstruct-2 cached SFT data
Check `/lustre/projects/polyullm/lipengxiang_tmp/dllm/.data/sft/llada/openmath2-500k/`.
The 1k cache subset will be drawn from here, so its existence is hard
required.

### 4.4 Storage budget for cache
Need ~25 GB free. `df -h /lustre/projects/polyullm/lipengxiang_tmp/dllm/.data/`.

**Acceptance:** a `phase0_status.md` file in the branch documenting
which of the above exist with their absolute paths and a short
GSM8K sanity number for the teacher.

---

## 5. Phase 1: Teacher Trajectory Cache

### 5.1 Deterministic confidence sampler
File: `dllm/core/samplers/mdlm_deterministic.py`.

Wraps `MDLMSampler.sample` but overrides the commit step to be
bit-exact reproducible:
- compute `softmax` in **fp32**;
- add a tiny per-position eps `-1e-7 * arange(L)` to the masked
  confidence to break ties deterministically (lower index wins);
- record the (logits, mask, committed) state at every step.

Public API:
```python
@dataclass
class DeterministicSamplerConfig(MDLMSamplerConfig):
    record_trajectory: bool = True
    top_k_logprobs: int = 64
    record_neg_log_tail_mass: bool = True

class DeterministicMDLMSampler(MDLMSampler):
    def sample_with_trajectory(self, inputs, config) -> SamplerWithTrajectoryOutput
```

`SamplerWithTrajectoryOutput` exposes `final_sequences` plus a
`trajectory: List[TrajectoryStepRecord]` per sample.

### 5.2 Cache record format
File: `dllm/pipelines/llada_looped/cache_io.py`.

```python
@dataclass
class TrajectoryStepRecord:
    step_idx: int                    # uint16
    mask_state_packed: bytes         # bit-packed mask, ceil(L_resp/8)
    top64_indices: np.ndarray        # int32  [L_supervised, 64]
    top64_logprobs: np.ndarray       # bf16   [L_supervised, 64]   normalised log p
    neg_log_tail_mass: np.ndarray    # bf16   [L_supervised]       = -log(1 - sum top64)
    teacher_top1_conf: np.ndarray    # bf16   [L_supervised]
    teacher_entropy: np.ndarray      # bf16   [L_supervised]
```

`L_supervised` = response positions only. Prompt slots are not stored.

Sharded layout (one *directory* per shard, one ``.npy`` per array key
inside; `np.savez` packs into a zip and won't memory-map for random
per-step access during training, so we don't use it):
```
.cache/teacher_traj/openmath2-1k-256step/
├── manifest.json
├── shard_0000/
│   ├── sample_ids.npy
│   ├── prompt_lens.npy
│   ├── final_sequences.npy
│   ├── mask_state_packed.npy
│   ├── top_k_indices.npy
│   ├── top_k_logprobs_bf16.npy
│   ├── neg_log_tail_mass_bf16.npy
│   ├── top1_conf_bf16.npy
│   └── entropy_bf16.npy
├── shard_0001/
│   └── ...
└── ...
```

Public API:
```python
def write_shard(path, shard: ShardData) -> None: ...
def read_shard(path, mmap=True, materialize_fp32=True) -> ShardData: ...   # tests / audit
def open_shard_arrays(path) -> dict[str, np.ndarray]: ...                  # training; np.memmap views
def CacheManifest.load(cache_dir, strict_git=False) -> CacheManifest: ...
```

### 5.3 Cache builder
File: `dllm/tools/build_teacher_cache.py`.

```bash
python -m dllm.tools.build_teacher_cache \
    --teacher_ckpt <path> \
    --dataset_path <openmath2-500k cache> \
    --num_samples 1000 \
    --teacher_steps 256 \
    --max_response_len 256 \
    --top_k 64 \
    --output_dir .cache/teacher_traj/openmath2-1k-256step \
    --shard_size 100 \
    --micro_batch_size 4
```

Determinism comes from ``temperature=0`` + ``stochastic_transfer=False``
hard-coded in :class:`DeterministicMDLMSampler`; no per-shard seed is
needed (and none is recorded). What goes into the manifest is the
sampler-config hash (``top_k``, ``teacher_steps``, ``tie_break_eps``, …)
and the git commit at build time (audit metadata only).

### 5.4 Mini-cache pre-flight (cheap, mandatory)
Before launching the full 1k build, run with `--num_samples 50`. Verify:
1. Storage extrapolation matches 25 GB target (within 30%).
2. Audit pass: in ≥75% of supervised positions, the cached top-64
   probability mass ≥ 0.90. If not, bump to `--top_k 128` and re-spec
   the storage budget.
3. Roundtrip: read back and recompute coarse KL against teacher
   logits in fp32; max abs error < 1e-3 in nats.

### 5.5 Phase 1 acceptance
- Full 1k cache built, manifest hashes recorded.
- Re-running on the same machine + seed re-produces a sampled shard
  bit-exactly.
- Mini-audit pass.

---

## 6. Phase 2: P0 Sanity (progressive distillation works on LLaDA)

Goal: verify trajectory distillation can reduce teacher 256 -> student
128 with < 2% GSM8K drop, **before** any inner-loop or re-entry
machinery exists. If P0 fails, the entire approach is brittle.

### 6.1 Loss
File: `dllm/core/trainers/td_losses.py`.

```python
def coarse_kl_with_tail(student_logits, teacher_top64_idx, teacher_top64_lp,
                        teacher_neg_log_tail, supervised_mask) -> Tensor:
```

Implementation in fp32 for the entire computation; clamp
`student_top64_mass` at `1 - 1e-6` before `log1p`.

### 6.2 Trainer
File: `dllm/core/trainers/td_distill.py`. Inherits
`transformers.Trainer` directly (not `MDLMTrainer`) because the input
unit is `(sample_id, student_step_idx)` from the cache, not online
masking.

```python
@dataclass
class TDDistillConfig(TrainingArguments):
    cache_dir: str = ...
    teacher_steps: int = 256
    student_steps: int = 128
    teacher_stride: int = 2     # auto = teacher_steps // student_steps
    top_k: int = 64
    # Phase 4/7 add T_rec_min/max, lambda_r, etc.
```

Custom collator: yields a batch of `(input_ids_at_step, supervised_mask,
teacher_target_step)` tuples by sampling student steps uniformly.

### 6.3 Training
- Optimizer AdamW lr=1e-5, wd=0.01, betas=(0.9, 0.95).
- 500-step linear warmup, then constant.
- Stride K=2 (teacher_step 0 -> student_step 0, teacher_step 2 ->
  student_step 1, ...).
- Train until held-out coarse KL plateaus (5 evals × <0.5%
  improvement, max 30k steps).

### 6.4 P0 acceptance
GSM8K accuracy at 128 student diffusion steps ≥ teacher@256 - 2%.
If gap > 5%, **stop and debug distillation pipeline; do not proceed
to inner-loop work**.

---

## 7. Phase 3: P0.5 Compressibility Diagnostics

File: `dllm/tools/compressibility_probe.py`.

Three diagnostics on the cached 1k trajectories. Output a single
`phase3_decision.json`. **No model training in this phase except the
two cheap probes below.**

### 7.1 Skip-probe predictability
Train a 2-layer MLP probe `belief(t) -> top-K log p(t+K)` for K ∈ {2,4,8}.
Held-out coarse KL vs the K=1 self-KL baseline.

### 7.2 Future-commit top-K inclusion
For commits made by teacher in [t+1, t+K_steps], what fraction were
already in step-t top-10? Report mean inclusion at K_steps ∈ {2,4,8}.

### 7.3 Velocity autocorrelation
`v_t = top1_logprob(t+1) - top1_logprob(t)`. Compute mean
`cosine(v_t, v_{t+1})` across supervised positions.

### 7.4 Decision rule

The cache is built on a fixed grid of ``teacher_steps=256`` and the
training dataset indexes by ``(sample, student_step)`` with stride
``teacher_steps / student_steps``. Hence ``student_steps`` must divide
``teacher_steps``; pick from the divisor set ``{32, 64, 128, 256}``
(``T_rec`` follows so the inner-loop budget per outer step is
``teacher_steps / student_steps``).

| K=4 skip-probe KL ratio | future-commit incl. @ K=4 | (student_steps, T_rec) |
|---|---|---|
| ≤ 1.5× K=1 self-KL | ≥ 0.7 | (64, 4) |
| 1.5–3× | 0.4–0.7 | (128, 2) |
| > 3× | < 0.4 | (128, 2) (still try, but expect P0.5 to flag low compressibility) |

If K=8 also viable: (32, 8). Acceptance: `phase3_decision.json` written
with the locked `(student_steps, T_rec)` and the underlying numbers.

---

## 8. Phase 4: P1 No-Reentry TD Baseline

File: `dllm/pipelines/llada_looped/models/modeling_no_reentry_td.py`.

```python
class NoReentryTD(LLaDALoopedModel):
    """T_rec inner R-passes per diffusion step. No re-entry. The only
    trainable module is `simple_residual_link`. Per-step coda decoding
    is enabled.
    """
    def __init__(self, config: NoReentryTDConfig):
        super().__init__(config)  # frozen prelude/R/coda/wte/ln_f
        self.simple_residual_link = SimpleResidualLink(d=config.d_model,
                                                        bottleneck=512)
    def forward(self, ..., return_per_step: bool = True) -> ...:
        # decodes coda(ln_f(h_r)) at every r, NOT only the last
```

`SimpleResidualLink`: LayerNorm + d->512->d residual, proj2 zero-init
(identity at init). Distinct class from V2.1's `RecursiveLink`.

### 8.1 Loss (per-step uniform)
```python
loss = mean_r [ coarse_kl_with_tail(student_logits_r, teacher_step_r) ]
```
where `teacher_step_r = teacher_trajectory[stride * student_step_idx + r]`.

### 8.2 Training
- Frozen everything in backbone; only `simple_residual_link.*` trains.
- AdamW lr=1e-3, wd=0.01 (probe-style LR; module is tiny).
- Stride and `(student_steps, T_rec)` from Phase 3.
- Logging: per-batch grid `(diffusion_step_quartile,
  mask_ratio_quantile)` of {`current_kl_r`, `next_kl_improvement`,
  `top1_conf_mean`, `teacher_entropy_mean`}.

### 8.3 Acceptance
- Held-out coarse KL plateaus.
- Report T_rec sweep at eval ∈ {1, 2, 3, 4}.
- **Critical question for Phase 7 viability**: does no-reentry TD
  reproduce V2.1's T=2 peak / T=4 drop? If yes, the inner loop is
  saturating early without re-entry, and re-entry is justified.
  If T_rec=4 already gives monotone gains here, the re-entry story
  weakens and Phase 7 should be re-scoped.

---

## 9. Phase 5: V2.1 Smoke-Swap (opportunistic, non-blocking)

Only runs if Phase 0 found V2.1 checkpoints.

File: `dllm/tools/transition_swap_eval.py`.

For a fixed reference `h_r` from V2.1:
- A: original `RecursiveLink` transition (V2.1's bridge).
- B: pure manifold re-entry (zero new params, no Δ).
- C: re-entry + tiny Δ probe trained locally on cached
  `(h_r, u_r, teacher_q_{r+1})` tuples (fp32 AdamW lr=1e-3, plateau).

Compare next-step coarse KL vs cached teacher target, bucketed by
`(diffusion_step_quartile, mask_ratio_bin, current_kl_quartile)`.

**Kill condition (strict)**: only if both V2.1-a and V2.1 fixed4 show
A wins B and C in **every** bucket -- only then short-circuit Phase 7.
Anything weaker is informational, not a kill.

---

## 10. Phase 6: P1.5 Transition Swap on P1's `h_r`

Same machinery as Phase 5, reference model is the P1 checkpoint from
Phase 4. P1's `h_r` is trajectory-aligned, which makes this the
**primary** commit gate for Phase 7.

Acceptance: a triple-bucketed report of A vs B vs C, with explicit
ordering per bucket. Output `phase6_decision.json` with `commit_p2:
true|false`.

---

## 11. Phase 7: P2 ReEntry-TD Full Model

File: `dllm/pipelines/llada_looped/models/modeling_reentry_td.py`.

```python
class ReentryTD(LLaDALoopedModel):
    """Manifold re-entry between inner steps. Trainable = correction_link.
    """
    def __init__(self, config: ReentryTDConfig):
        super().__init__(config)
        self.correction_link = CorrectionLink(d=config.d_model,
                                               bottleneck=512,
                                               max_steps=config.max_t_rec)

    def forward(self, ..., reentry_pattern='every-step'):
        # for r in range(T_rec):
        #     logits_r = lm_head(ln_f(coda(h)))
        #     if r < T_rec-1 and should_reenter(r+1):
        #         p_r = softmax(logits_r); soft = top_k_soft_emb(p_r, K=64)
        #         u_r = prelude(soft)               # uses input_embeddings path
        #         delta = correction_link(h, u_r, step_idx=r)
        #         h = R(u_r + delta)
        #     else:
        #         h = R(h)
```

### 11.1 `CorrectionLink`
LayerNorm `h`, `u`, `du = u - h`; projection to bottleneck; sum +
`step_emb`; GELU; `out` projection zero-init. **RMS cap**:
`||delta||_RMS ≤ 0.2 * ||u||_RMS` per token.

### 11.2 Soft top-K embedding
On masked positions only: take top-K probs from `p_r`, renormalise
within top-K, weighted sum of `wte` rows. Prompt positions keep their
hard embedding.

### 11.3 Patterns
`reentry_pattern ∈ {none, one-shot, two-shot, every-step}`. Default
P2 first run is `every-step`.

### 11.4 Acceptance
- Trains, plateau on held-out coarse KL.
- T_rec sweep report.
- Compare GSM8K and next-step KL vs P1 at the same `(student_steps,
  T_rec)` config.

---

## 12. Phase 8: Sweep / Selective Inner Loop

Data-driven only. Two axes:
- **Axis 1** -- `reentry_pattern` ∈ {none, one-shot, two-shot,
  every-step} at fixed `(student_steps, T_rec)`. Plot quality vs FLOPs
  Pareto.
- **Axis 7** -- mask-ratio-conditional inner loop. Implement only if
  Phase 4/7 logging shows the marginal-gain map is highly non-uniform.

---

## 13. Numerical Hazards (do not skip)

1. **Cache I/O bf16; KL computation fp32.** Every entry point that
   reads cached `top_k_logprobs_bf16` or `neg_log_tail_mass_bf16` must
   cast to fp32 before any `exp` / `log` / sum. ``coarse_kl_with_tail``
   raises if it gets a non-fp32 student logits tensor; the dataset
   does the cast in ``__getitem__`` so the collator and loss never see
   the uint16 view.
2. **Tail mass storage**: `neg_log_tail_mass = -log(1 - sum_top_k)`.
   Reverse: `tail_mass = exp(-neg_log_tail_mass)`. Underflow at very
   confident steps is acceptable -- KL contribution becomes 0.
3. **Tie-break in fp64.** The deterministic sampler promotes confidence
   to fp64 for the commit-selection ``topk`` and adds a per-position
   ``-1e-12 * pos_idx`` offset. The eps is small enough that for
   fp32-distinct confidences ordering is preserved (``eps * T`` stays
   well below fp32 precision). The straight ``-1e-7 * pos`` shape from
   an earlier draft is wrong: at T=1024 it perturbs by 1e-4, which
   *does* reorder fp32-distinct positions.
4. **Coarse KL**: clamp `student_top_k_mass` at `1 - 1e-6` before
   `log1p` -- otherwise NaN gradient at confident student.
5. **Cache compatibility stamp**. Manifest records
   ``sampler_config_hash`` (top_k, teacher_steps, tie_break_eps, …)
   and ``git_commit_hash``. The hash that *determines* cache content
   is the sampler hash; the git stamp is audit metadata.
   ``CacheManifest.load`` warns on git drift and only raises when
   ``strict_git=True``. The trainer config exposes ``--strict_git``
   for runs that want hard pinning. There is no ``--allow_stale``;
   that contract was removed because the default behaviour is now
   already permissive.

---

## 14. Tests (`tests/td/`)

Each test is a single file, runnable via
`python -m pytest tests/td/test_X.py`.

| File | Asserts |
|---|---|
| `test_sampler_deterministic.py` | Same seed + same hardware -> bit-exact next-step commit |
| `test_coarse_kl.py` | At K=V (full vocab), coarse KL matches full softmax KL within 1e-6 |
| `test_reentry_zero_init.py` | `CorrectionLink.delta == 0` at init -> `h_2 == R(u_1)` |
| `test_norm_cap.py` | `||delta||_RMS / ||u||_RMS ≤ 0.2` for random h, u inputs |
| `test_cache_roundtrip.py` | write → read preserves all fields; KL recomputed from cache matches reference within 1e-3 nats |

---

## 15. Recommended schedule

| Week | Phase | GPU need |
|---|---|---|
| 1 | Phase 0 + Phase 1 mini-cache + 1k cache build | 4-8 GPU-days |
| 2 | Phase 2 (P0) + Phase 3 (P0.5) + opportunistic Phase 5 | 8-16 GPU-days |
| 3 | Phase 4 (P1) | 8-16 GPU-days |
| 4 | Phase 6 (P1.5) decision | 1 GPU-day |
| 5 | Phase 7 (P2) | 16-24 GPU-days |
| 6+ | Phase 8 sweep | 16+ GPU-days |

Phase boundaries are hard gates: do not start the next phase until
the previous phase's acceptance criterion is satisfied.

---

## 16. References inside this repo

- V2.1 looped overview: `/Users/pixeli/dllm/LOOPED_LLADA_RESEARCH.md`
- V2.1 model: `/Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py`
- V2.1 trainer: `/Users/pixeli/dllm/dllm/core/trainers/mdlm_looped.py`
- Base sampler being reused/wrapped: `/Users/pixeli/dllm/dllm/core/samplers/mdlm.py`
- Repo navigation (trust this over README): `/Users/pixeli/dllm/AGENTS.md`
