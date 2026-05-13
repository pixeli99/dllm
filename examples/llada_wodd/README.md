# LLaDA-WoDD: Write-on-Demand Diffusion

**WoDD** is a minimal architectural change to a masked diffusion language
model (e.g. LLaDA) that gives it an *append-only* latent scratchpad which
the model fills on demand during the denoising forward.

## Why

All existing latent reasoning mechanisms for DLMs — **MetaState**, **LATTS**,
**DPad**, **Latent Tokens**, **LaDiR** — share one structural property: they
*evolve* a fixed-capacity state (GRU memory, recurrent hidden, suffix scratch,
VAE block). Empirically they cap at **~5 %** lift on math reasoning while
token CoT scales **30–50 %**. Hypothesis: this is the information-theoretic
ceiling. Token CoT writes `log|V|` new bits each step (append-only); state
evolution recycles a fixed buffer.

WoDD restores append-only semantics in latent space. The model decides
*per inner step* whether to write a vector to a tape; subsequent inner
steps cross-attend into the tape. Tape entries are never overwritten.

## Mechanism

One denoising forward pass:

```text
e   = Embed(masked_input)
h   = prelude_blocks(e)
M   = []                                      # tape
for k in 1..T_step:
    if M non-empty: h <- h + TapeRead(h, M)
    h <- recurrent_blocks(h)                  # weight-shared across k
    pool = masked_mean(h)
    g_k, v_k = WriteHead(pool)
    M.append(g_k * v_k)                       # APPEND ONLY
x   = coda_blocks(h)
logits = LMHead(ln_f(x))
```

Soft gating (`g_k * v_k` before append) preserves append-only semantics
while keeping gradients clean — no ST-Gumbel needed. A near-zero `g_k`
means the entry contributes ~0 to subsequent cross-attention.

## What WoDD is *not* (defends novelty)

| Method | What it does | Why it isn't WoDD |
|---|---|---|
| MetaState | 64×1024 GRU memory updated in place | State evolution, not append-only |
| LATTS | More latent steps on same state | No memory at all |
| DPad | 256 pre-allocated suffix scratch | Fixed, not written on demand |
| Latent Tokens | Pre-inserted masked positions | Hyperparameter count, not learned |
| LaDiR | Latent diffusion module on frozen LLM | Separate paradigm |
| Coconut | AR hidden feeds back into next input | AR, no DLM, no append tape |
| ReMDM | Re-mask already unmasked tokens | Mutates output, not separate scratch |

## Files

```
dllm/pipelines/llada_wodd/
  models/
    configuration_llada_wodd.py    # LLaDAWoDDConfig (+ WoDD knobs)
    _wodd_modules.py               # WriteHead, TapeRead, MemoryTape
    modeling_llada_wodd.py         # LLaDAWoDDModel(LM) — forward w/ inner loop
dllm/core/trainers/
  mdlm_wodd.py                     # MDLMWoDDTrainer w/ sparsity + entropy loss
examples/llada_wodd/sft.py         # SFT entry
scripts/llada_wodd/run.sh          # 1-node FSDP launch
tests/wodd/
  test_wodd_modules.py             # 16 module unit tests
  test_wodd_invariants.py          # 5 invariant tests (zero-init, append-only, grad)
  test_wodd_forward.py             # 6 toy-model forward tests (skips if no lm_eval)
```

## Invariants verified by tests

1. **Zero-init drop-in**: `zero_init_wodd=True` makes step-0 forward bit-
   identical to a vanilla split LLaDA (T_step folds of the recurrent block
   on the same input). Proved by `test_zero_init_degenerates_to_pure_weight_sharing`.
2. **Append-only**: `MemoryTape` exposes no `.overwrite / .pop / .__setitem__`.
   Proved by `test_memory_tape_append_only`, `test_append_only_across_iterations`.
3. **Gradient flows across all iterations**: BPTT through T_step inner loops
   reaches the first iteration's write. Proved by `test_grad_flow_across_all_iterations`.
4. **Sparsity pressure works**: a pure `mean(g)` loss drives the gate bias
   down. Proved by `test_gate_sparsity_pressure_works`.

## Usage

```bash
bash scripts/llada_wodd/run.sh
```

Key TensorBoard scalars to watch:
- `wodd/mean_gate` — should rise above the bias-driven init (`sigmoid(-2) ≈ 0.12`)
  for hard examples once the model finds writes useful.
- `wodd/gate_entropy` — should stay above ~0.2 (no collapse).
- `wodd/loss_mdlm` — should match a frozen split-LLaDA at step 0
  (zero-init invariant) and improve with training.
- `wodd/tape_contrib_rel_norm_iter{k}` — relative norm of tape cross-attn
  output vs h. Starts at 0; growing means the tape is being used.

## Falsifiable predictions (W1–W8)

1. **Iso-FLOP reasoning gain** ≥ 3 pp on GSM8K vs MetaState (matched compute).
2. **`mean_gate` correlates with AR-CoT trace length** at ρ ≥ 0.6 across
   problem difficulty bands.
3. **Disabling the tape** (force `g_k = 0`) at inference should drop
   accuracy proportional to problem difficulty.
4. **Probing tape contents** — linear probe from `v_k` to AR-CoT step-k
   token class should beat chance significantly.
5. **Capacity scaling law**: sweep `tape_dim ∈ {64, 256, 1024, 4096}`,
   accuracy should rise monotonically until compute-bound. (This is the
   capacity-bottleneck test of WoDD vs MetaState's 64×1024 plateau.)

If (1) fails, WoDD is no better than MetaState — falsifies the append-only-
is-the-missing-mechanism hypothesis. That is the primary go/no-go.
