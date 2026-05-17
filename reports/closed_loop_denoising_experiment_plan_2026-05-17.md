# Closed-Loop Denoising 下一步实验计划

更新时间：2026-05-17 HKT
本文件对应代码分支：`lpx/loop_v2_1a`
本地仓库：`/Users/pixeli/dllm`
集群仓库：`/lustre/projects/polyullm/lipengxiang_tmp/dllm`

## 0. 当前结论

当前最强结果不是 workspace，也不是 deep supervision，而是 **decayed damping / step-size schedule**：

| 方法 | 最佳 T_rec | GSM8K limit=320 Math-Verify | 正确数 |
|---|---:|---:|---:|
| constant alpha=0.5 | 3 | 0.815625 | 261 / 320 |
| no-workspace normalized tau=1.0 | 2 | 0.821875 | 263 / 320 |
| no-workspace decay beta=0.5 | 6 | 0.853125 | 273 / 320 |
| no-workspace decay beta=1.0 | 5 | 0.809375 | 259 / 320 |
| workspace s8 normalized | 2 / 3 | 0.818750 | 262 / 320 |
| workspace s16 normalized | 2 | 0.818750 | 262 / 320 |
| workspace s32 normalized | 1 | 0.790625 | 253 / 320 |

研究表述应该从 “damping looped LLaDA” 改成：

> Closed-Loop Denoising: turning an open pretrained denoising segment into a stable recurrent self-map.

核心公式：

```text
h_1 = R(P(x_t))
G(h) = R(L(h))
h_{r+1} = h_r + alpha_r * (G(h_r) - h_r)
logits = C(h_R)
```

`RecursiveLink L` 解决 representation closure：把原本 `R: H_in -> H_out` 的 open segment 变成 `G: H_out -> H_out` 的 self-map。
`damping / alpha_r` 不是普通 trick，而是 under-relaxed fixed-point iteration 的 step size。

## 1. 新代码实现：Diffusion-Conditioned Decayed Damping

新增方法名：

```text
Diffusion-Guided Budgeted Latent Refinement
```

新增训练脚本：

```text
/lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/run_diffusion_conditioned_decay.sh
```

新增/修改代码：

| 文件 | 作用 |
|---|---|
| `/Users/pixeli/dllm/dllm/pipelines/llada_looped/models/configuration_llada_looped.py` | 增加 diffusion-conditioned damping 配置 |
| `/Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py` | 在 feedback update 里加入 mask-ratio tau 和 token gate |
| `/Users/pixeli/dllm/dllm/core/trainers/mdlm_looped.py` | 新参数进入 loop optimizer group，并记录诊断指标 |
| `/Users/pixeli/dllm/examples/llada_looped/sft.py` | 暴露 CLI 参数，并把 tokenizer 的 `mask_token_id` 写入模型 config |
| `/Users/pixeli/dllm/scripts/looped_llada/run_diffusion_conditioned_decay.sh` | 主实验启动脚本 |

新方法只在 `r >= 1` 的 feedback pass 生效，因此 `T_rec=1` 仍然保持 vanilla split LLaDA 路径。

具体 update：

```text
alpha_{b,r,i} = tau_b * softmax(-beta * r) * gate_i

tau_b = softplus(tau_logit + slope * (mask_ratio_b - ref))

gate_i =
  1.0            if token_i is <mask>
  unmasked_gate  otherwise
```

默认初始化：

```text
beta = 0.5
tau = 1.0
mask_tau_ref = 0.5
mask_tau_slope_init = 0.0
damping_masked_gate = 1.0
damping_unmasked_gate = 0.1
learn_damping_tau = True
learn_mask_tau = True
T_rec train range = 2..6
```

由于 `mask_tau_slope_init=0.0`，训练开始时等价于当前 winner 的 fixed decay schedule，再由 stage1 训练学习是否根据 mask ratio 调整总 refinement budget。

新增诊断项：

```text
loop/mask_ratio
loop/damping_tau
loop/mask_tau_slope
loop/damping_alpha_iter*
loop/damping_alpha_effective_iter*
loop/damping_alpha_sum
loop/damping_alpha_effective_sum
loop/damping_masked_gate
loop/damping_unmasked_gate
```

## 2. 实验 A：主方法训练

目标：验证 DLM 的 mask/noise state 是否能进一步提升固定 `decay beta=0.5`。

集群命令：

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /lustre/projects/polyullm/lipengxiang_tmp/dllm

BETA=0.5 \
TAU=1.0 \
UNMASKED_GATE=0.1 \
MASK_TAU_REF=0.5 \
MASK_TAU_SLOPE_INIT=0.0 \
T_REC_MIN=2 \
T_REC_MAX=6 \
T_REC_EVAL=6 \
MAX_STEPS=1000 \
bash /lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/run_diffusion_conditioned_decay.sh
```

预期输出目录：

```text
/lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-v2.1a-stage1-ins-diffcond-decay-tau1.0-b0.5-ug0.1-t2-6/checkpoint-final
```

训练设置：

| 项 | 值 |
|---|---:|
| base model | `/lustre/projects/polyullm/lipengxiang_tmp/LLaDA-8B-Instruct` |
| data | `/lustre/projects/polyullm/lipengxiang_tmp/dllm/.data/sft/llada/openmath2-500k` |
| max steps | 1000 |
| max length | 1024 |
| per-device batch | 4 |
| grad accumulation | 4 |
| lr | 5e-4 |
| loop lr mult | 1.0 |
| train T_rec | 2..6 |
| default eval T_rec | 6 |
| backbone freeze | freeze prelude, R, coda, ln_f, wte |
| trainable params | `recursive_link.*`, `model.damping_tau_logit`, `model.mask_tau_slope` |

验收标准：

| 结果 | 判断 |
|---|---|
| best >= 276 / 320 | 明确强于 fixed decay，可作为下一版主方法 |
| best = 273±2 / 320 且 T4/T6/T8 稳定 | 方法可保留；主要贡献是稳定性而非峰值 |
| best < 268 / 320 | mask-conditioned gate/tau 当前失败，先回到 fixed decay 主线 |

必须检查：

1. `T_rec=1` 是否仍在 vanilla 附近，即 Math-Verify 约 `253 / 320`。
2. `loop/mask_tau_slope` 是否真的更新，不应一直精确为 0。
3. `loop/damping_alpha_effective_sum` 是否明显小于 `loop/damping_alpha_sum`，否则 token gate 没有生效。
4. `T_rec=4/6/8` 是否比 fixed decay 更平滑，而不是只在一个 T 上偶然跳高。

## 3. 实验 B：主方法评测

目标：只用 limit=320 先看信号，不做全量矩阵。

建议评测 T：

```text
T_rec = 1, 2, 4, 6, 8
```

顺序评测命令：

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /lustre/projects/polyullm/lipengxiang_tmp/dllm

ckpt="/lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-v2.1a-stage1-ins-diffcond-decay-tau1.0-b0.5-ug0.1-t2-6/checkpoint-final"

for t in 1 2 4 6 8; do
  bash /lustre/projects/polyullm/lipengxiang_tmp/dllm/scripts/looped_llada/eval_gsm8k.sh \
    --checkpoint "${ckpt}" \
    --t_rec "${t}" \
    --num_gpu 8 \
    --batch_size 4 \
    --num_fewshot 0 \
    --max_new_tokens 256 \
    --steps 256 \
    --block_size 64 \
    --cfg_scale 0.0 \
    --limit 320
done
```

重要注意：

- `eval_gsm8k.sh --t_rec` 会原地 patch `config.json` 的 `mu_rec_eval`。
- 并行评测不同 T 时不要直接共用同一个 checkpoint 目录；要先创建 per-T checkpoint view，或者顺序跑。
- GSM8K 主指标使用 Math-Verify，不用 lm-eval strict-match 判断模型质量。

Math-Verify 命令使用当前集群已有脚本：

```bash
python /lustre/projects/polyullm/lipengxiang_tmp/dllm/eval_math_verify_jsonl.py \
  --samples_jsonl "${ckpt}/eval_gsm8k_T6/samples_gsm8k_cot_*.jsonl" \
  --output_dir "${ckpt}/eval_gsm8k_T6/math_verify" \
  --keep-filter flexible-extract \
  --dedupe-filter-rows
```

每个 T 都生成：

```text
eval_gsm8k_T*/results_*.json
eval_gsm8k_T*/samples_gsm8k_cot_*.jsonl
eval_gsm8k_T*/math_verify/summary_T*.txt
eval_gsm8k_T*/math_verify/scored_T*.jsonl
eval_gsm8k_T*/math_verify/wrong_T*.jsonl
```

## 4. 实验 C：固定 decay winner 的动力学诊断

目标：不是再跑训练，而是证明 fixed decay beta=0.5 为什么赢。

需要比较的 checkpoint：

```text
/lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-v2.1a-stage1-ins-damped-a0.5/checkpoint-final
/lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-v2.1a-stage1-ins-damped-norm-tau1.0/checkpoint-final
/lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-v2.1a-stage1-ins-damped-decay-tau1.0-b0.5/checkpoint-final
/lustre/projects/polyullm/lipengxiang_tmp/dllm/.models/loop_belief/openmath2-v2.1a-stage1-ins-damped-decay-tau1.0-b1.0/checkpoint-final
```

要看的 TensorBoard scalar：

```text
loop/residual_norm_iter*
loop/raw_residual_norm_iter*
loop/adapter_update_norm_iter*
loop/damping_alpha_iter*
loop/damping_tau
loop/damping_decay_beta
```

要回答的问题：

1. `decay beta=0.5` 的 residual 是否随 iteration 更稳定下降？
2. normalized 是否在后期 residual 不降或震荡？
3. constant alpha=0.5 是否只是把有效最佳点从 T=2 平移到 T=3？
4. `beta=1.0` 是否前期步长过大、后期 correction 不够？

接受标准：

- 如果 `beta=0.5` 的 `residual_norm_iter*` 更像收缩轨迹，可以把方法解释成 stable fixed-point denoising。
- 如果 residual 不收缩但 accuracy 仍提升，需要改写 claim：它是 budgeted refinement，而不是严格 contraction。

## 5. 实验 D：workspace 只做一发完整体检查

这不是主线，只用于回答 “workspace 和 winner schedule 是否有协同”。

只跑一个设置：

```text
workspace s8 + decay beta=0.5 + learnable tau
```

如果要跑，建议从现有 `run_workspace.sh` 派生，而不是继续扫 s16/s32。

建议设置：

```text
SLOTS=8
BETA=0.5
TAU=1.0
T_REC_MIN=2
T_REC_MAX=6
ENABLE_DEEP_SUP=False
```

验收标准：

| 结果 | 判断 |
|---|---|
| best >= 277 / 320 | workspace 与 decay 有协同，可以重新考虑 workspace story |
| best = 273±2 / 320 | workspace 不伤害但不提供主要增益 |
| best < 268 / 320 | workspace 继续降级，不再投入 |

当前不建议继续做：

- s16/s32 sweep
- deep supervision lambda sweep
- ExitAdapter
- 普通 time embedding / gate

## 6. 下一阶段方法方向

如果实验 A 成功，下一版方法就写成：

```text
Closed-Loop Denoising
+ Diffusion-Guided Step-Size Control
```

如果实验 A 持平但不提升，保留 fixed decay beta=0.5 作为主方法，下一步做 residual-aware controller：

```text
if residual_ratio rises or delta_cos < 0:
    alpha_next *= shrink
```

如果实验 A 明显失败，回到 fixed decay beta=0.5，并把 novelty 聚焦为：

```text
representation closure + damped fixed-point iteration
```

## 7. 论文里该怎么写

不要写：

```text
We add damping to looped LLaDA.
```

应该写：

```text
A pretrained transformer segment is an open map between layer distributions.
We close it into a self-map with a zero-initialized RecursiveLink, then solve
the induced denoising fixed point with a budgeted recurrent step-size schedule.
DLM mask/noise state controls where and how much latent refinement is applied.
```

当前最强 claim：

1. naive recurrence fails because total latent update grows with depth;
2. representation closure makes the segment self-iterable;
3. damped fixed-point integration prevents overshoot;
4. diffusion-conditioned step-size control makes the solver DLM-specific.
