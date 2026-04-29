# Markdown Guidelines

- Use **absolute paths** instead of relative paths.

# Python Guidelines

- At the top of each file, include a **docstring** with simple instructions on how to run the code.
- When writing new code: preview existing code first, reuse existing modules where possible, and keep the new code’s style consistent with the codebase.
- Before running scripts: source `~/.zshrc` and activate conda env `dllm` (e.g. `conda activate ~/miniconda3/envs/dllm`).
- For tasks requiring a GPU, use the following command: `srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python ...`.

# Repo Quick Reference

This repository is a diffusion-language-model training/inference library. For LLaDA work, do not start from a random model file. Start from the entry scripts and shared trainer/sampler utilities below.

## Environment

- Local workspace repo root: `/Users/pixeli/dllm`
- Cluster repo root: `/lustre/projects/polyullm/lipengxiang_tmp/dllm`
- This project is run on the cluster; when giving cluster commands, scripts,
  checkpoint paths, eval paths, or Slurm examples, use
  `/lustre/projects/polyullm/lipengxiang_tmp/dllm` as the repo root.
- Install path expectation on the cluster:
  `pip install -e /lustre/projects/polyullm/lipengxiang_tmp/dllm`
- Local install path expectation, only for local checks:
  `pip install -e /Users/pixeli/dllm`
- Before running training or tools, use:

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /lustre/projects/polyullm/lipengxiang_tmp/dllm
```

- If editable install is missing, a temporary fallback is:

```bash
export PYTHONPATH=/lustre/projects/polyullm/lipengxiang_tmp/dllm:${PYTHONPATH}
```

## Read This First

- Global overview: `/Users/pixeli/dllm/README.md`
- LLaDA usage doc: `/Users/pixeli/dllm/examples/llada/README.md`
- LLaDA SFT entry: `/Users/pixeli/dllm/examples/llada/sft.py`
- LLaDA pretraining entry: `/Users/pixeli/dllm/examples/llada/pt.py`
- LLaDA sampling entry: `/Users/pixeli/dllm/examples/llada/sample.py`
- LLaDA chat entry: `/Users/pixeli/dllm/examples/llada/chat.py`
- Shared masked-diffusion trainer: `/Users/pixeli/dllm/dllm/core/trainers/mdlm.py`
- Shared masked-diffusion sampler: `/Users/pixeli/dllm/dllm/core/samplers/mdlm.py`
- Model/tokenizer loading hooks: `/Users/pixeli/dllm/dllm/utils/models.py`
- Shared CLI dataclasses: `/Users/pixeli/dllm/dllm/utils/configs.py`
- Dataset loading/parsing: `/Users/pixeli/dllm/dllm/data/utils.py`
- Dataset post-processing/tokenization: `/Users/pixeli/dllm/dllm/utils/data.py`
- Slurm launcher: `/Users/pixeli/dllm/scripts/train.slurm.sh`

## LLaDA Train Flow

### 1. SFT

Main path:

`/Users/pixeli/dllm/examples/llada/sft.py`
→ `dllm.utils.get_model()`
→ `dllm.utils.get_tokenizer()`
→ `dllm.data.load_sft_dataset()`
→ `dllm.utils.default_sft_map_fn()`
→ `dllm.utils.post_process_dataset()`
→ `dllm.core.trainers.MDLMTrainer`

Important details:

- Default base model is `GSAI-ML/LLaDA-8B-Base`.
- `dataset_args` defaults to `allenai/tulu-3-sft-mixture[train:10000,test:1000]`.
- `default_sft_map_fn()` expects each row to contain `messages`.
- `mask_prompt_loss=True` means prompt tokens are labeled `-100`; only response tokens contribute loss.
- The collator is wrapped by `dllm.utils.NoAttentionMaskWrapper(...)` so padded EOS remains visible to the model.
- Final model is explicitly saved to `checkpoint-final`.

### 2. Pretraining

Main path:

`/Users/pixeli/dllm/examples/llada/pt.py`
→ `transformers.AutoConfig.from_pretrained()`
→ `transformers.AutoModel.from_config(..., init_params=True)`
→ `dllm.utils.get_tokenizer()`
→ `dllm.utils.load_peft()`
→ `dllm.data.load_pt_dataset(..., streaming=True)`
→ `dllm.utils.tokenize_and_group()`
→ `dllm.core.trainers.MDLMTrainer`

Important details:

- PT initializes weights from config, not from pretrained checkpoint weights.
- `model_name_or_path` is used mainly as a config/template source for the architecture.
- Default PT dataset is `mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]`.
- PT uses streaming by default.
- `tokenize_and_group()` concatenates text and chunks it into fixed-length sequences.
- `RandomTruncateWrapper` is used during PT with `random_length_ratio` to randomly shorten some batches.

### 3. RL / GRPO

The active LLaDA RL entry is not under `/Users/pixeli/dllm/examples/llada`.

Use:

- RL train entry: `/Users/pixeli/dllm/examples/rl/grpo/llada/train.py`
- RL trainer: `/Users/pixeli/dllm/dllm/pipelines/rl/grpo/trainer.py`

Important details:

- RL uses `DiffuGRPOTrainer`, not `MDLMTrainer`.
- It builds an `MDLMSampler` and uses denoising generation inside GRPO.
- LoRA for RL is configured separately via `trl`/`peft`, not through `dllm.utils.get_model(..., lora=True)`.

## Key Shared Abstractions

### `dllm.utils.get_model`

File: `/Users/pixeli/dllm/dllm/utils/models.py`

- Preferred model loader for most tasks.
- Tries `AutoModelForMaskedLM.from_pretrained(...)`, then falls back to `AutoModel.from_pretrained(...)`.
- Supports 4-bit loading through bitsandbytes.
- Applies LoRA via `dllm.utils.load_peft()` when `model_args.lora=True`.

### `dllm.utils.get_tokenizer`

File: `/Users/pixeli/dllm/dllm/utils/models.py`

- Preferred tokenizer loader for all LLaDA work.
- It mutates tokenizer special tokens and chat template based on the resolved model class.
- For LLaDA base, it adds `mask_token = "<|mdm_mask|>"`.
- For LLaDA MoE / LLaDA2 / LLaDA2.1 MoE variants, it adds `mask_token = "<|mask|>"`.
- It also patches chat templates, so using raw `AutoTokenizer` can silently break formatting.

### `MDLMTrainer`

File: `/Users/pixeli/dllm/dllm/core/trainers/mdlm.py`

- This is the core masked-diffusion training loop.
- On each batch it samples a timestep `t`, converts that to masking probability with the scheduler, masks tokens, runs the model once, and computes token-level cross-entropy on masked positions only.
- Scheduler defaults to `LinearAlphaScheduler`.
- Loss weighting and normalization are configurable in `MDLMConfig`.

### `MDLMSampler`

File: `/Users/pixeli/dllm/dllm/core/samplers/mdlm.py`

- This is the main inference/sampling implementation for LLaDA-style models.
- Generation is iterative denoising over appended mask tokens.
- Key knobs are `steps`, `block_size`, `temperature`, `remasking`, and `cfg_scale`.
- `sample()` handles normal generation; `infill()` fills explicit mask tokens in-place.

## Dataset Conventions

Files:

- `/Users/pixeli/dllm/dllm/data/utils.py`
- `/Users/pixeli/dllm/dllm/utils/data.py`

Important details:

- `dataset_args` supports forms like:
  - `"tatsu-lab/alpaca"`
  - `"allenai/tulu-3-sft-mixture[train:5000,test:1000]"`
  - `"dataset_a + dataset_b"`
  - `"OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python]"`
- SFT loaders normalize to `DatasetDict` and can concatenate multiple datasets.
- PT loader supports both streaming and non-streaming paths.
- `post_process_dataset()` either filters or truncates to `max_length`.
- For SFT rows with `prompt_len`, right truncation preserves the prompt and clips the response budget.

## Model Registration / Why `import dllm` Matters

Files:

- `/Users/pixeli/dllm/dllm/pipelines/llada/models/__init__.py`
- `/Users/pixeli/dllm/dllm/pipelines/llada2/models/__init__.py`
- `/Users/pixeli/dllm/dllm/pipelines/llada21/models/__init__.py`

Important details:

- `import dllm` triggers imports that register custom configs/models with Hugging Face Auto classes.
- For LLaDA, `model_type="llada"` and `model_type="lladamoe"` are registered there.
- If a checkpoint is MoE but its `config.json` still says `llada`, loading can fail or pick the wrong class.
- For MoE checkpoints, verify `config.json` if loading behaves strangely.

## Actual Commands Worth Reusing

### Quick SFT test on 1 GPU

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /Users/pixeli/dllm
accelerate launch \
  --config_file /Users/pixeli/dllm/scripts/accelerate_configs/ddp.yaml \
  --num_processes 1 \
  /Users/pixeli/dllm/examples/llada/sft.py \
  --load_in_4bit True \
  --lora True
```

### Standard SFT on 8 GPUs

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /Users/pixeli/dllm
accelerate launch \
  --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
  /Users/pixeli/dllm/examples/llada/sft.py
```

### Preprocess SFT data for multi-node reuse

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /Users/pixeli/dllm
python /Users/pixeli/dllm/dllm/tools/preprocess_sft_dataset.py \
  --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
  --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
  --dataset_args "allenai/tulu-3-sft-mixture" \
  --output_dir "/Users/pixeli/dllm/.data/sft/llada/tulu-3-sft-mixture" \
  --num_proc 64
```

### Pretraining

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /Users/pixeli/dllm
accelerate launch \
  --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
  /Users/pixeli/dllm/examples/llada/pt.py
```

### Sampling

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /Users/pixeli/dllm
python /Users/pixeli/dllm/examples/llada/sample.py \
  --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```

### Evaluation

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /Users/pixeli/dllm
accelerate launch --num_processes 4 \
  /Users/pixeli/dllm/dllm/pipelines/llada/eval.py \
  --tasks "gsm8k_cot" \
  --model "llada" \
  --apply_chat_template \
  --num_fewshot 5 \
  --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"
```

### Slurm

- First create logs dir: `/Users/pixeli/dllm/.logs`
- Update partition/quotatype in `/Users/pixeli/dllm/scripts/train.slurm.sh`
- The launcher forwards all args after the first unknown option to the Python script.

Example:

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /Users/pixeli/dllm
sbatch --gres=gpu:8 /Users/pixeli/dllm/scripts/train.slurm.sh \
  --accelerate_config "fsdp" \
  --script_path "/Users/pixeli/dllm/examples/llada/sft.py" \
  --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
  --dataset_args "tatsu-lab/alpaca" \
  --output_dir "/Users/pixeli/dllm/.models/LLaDA-8B-Base/alpaca"
```

## If You Need To Change Behavior, Edit Here

- Change diffusion loss/training math:
  `/Users/pixeli/dllm/dllm/core/trainers/mdlm.py`
- Change diffusion sampling behavior:
  `/Users/pixeli/dllm/dllm/core/samplers/mdlm.py`
- Change scheduler behavior:
  `/Users/pixeli/dllm/dllm/core/schedulers/`
- Change LLaDA tokenizer special tokens or chat template:
  `/Users/pixeli/dllm/dllm/utils/models.py`
- Change SFT/PT data loading syntax:
  `/Users/pixeli/dllm/dllm/data/utils.py`
- Change sequence clipping/grouping logic:
  `/Users/pixeli/dllm/dllm/utils/data.py`
- Change LLaDA eval harness defaults:
  `/Users/pixeli/dllm/dllm/pipelines/llada/eval.py`
- Change Slurm launch behavior:
  `/Users/pixeli/dllm/scripts/train.slurm.sh`

## Known Doc Drift

- `/Users/pixeli/dllm/examples/llada/README.md` mentions `/Users/pixeli/dllm/dllm/pipelines/llada/trainer.py`, but the shared trainer actually lives at `/Users/pixeli/dllm/dllm/core/trainers/mdlm.py`.
- `/Users/pixeli/dllm/examples/llada/README.md` mentions `examples/llada/grpo.py`, but the active RL entry is `/Users/pixeli/dllm/examples/rl/grpo/llada/train.py`.
- When in doubt, trust the actual Python entry scripts over the README file tree blocks.

## Fast Decision Rules For Future Agents

- If the task is “how does LLaDA train here?”:
  start with `/Users/pixeli/dllm/examples/llada/sft.py`, `/Users/pixeli/dllm/examples/llada/pt.py`, and `/Users/pixeli/dllm/dllm/core/trainers/mdlm.py`.
- If the task is “why is loading/tokenizer broken?”:
  inspect `/Users/pixeli/dllm/dllm/utils/models.py` first.
- If the task is “dataset_args syntax / preprocessing / truncation?”:
  inspect `/Users/pixeli/dllm/dllm/data/utils.py` and `/Users/pixeli/dllm/dllm/utils/data.py`.
- If the task is “sampling/eval behavior looks wrong?”:
  inspect `/Users/pixeli/dllm/dllm/core/samplers/mdlm.py` and `/Users/pixeli/dllm/dllm/pipelines/llada/eval.py`.
- If the task is “multi-node launch doesn’t work?”:
  inspect `/Users/pixeli/dllm/scripts/train.slurm.sh` and the YAML files in `/Users/pixeli/dllm/scripts/accelerate_configs/`.
