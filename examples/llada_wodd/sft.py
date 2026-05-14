"""LLaDA-WoDD SFT entry.

Retrofits a pretrained LLaDA-8B checkpoint with WoDD (Write-on-Demand
Diffusion) and trains under MDLMWoDDTrainer with:

  L = L_mdlm + λ_sparsity * mean(g_k) - λ_entropy * mean(H(Bern(g_k)))

Defaults freeze the vanilla LLaDA backbone (prelude + recurrent + coda +
ln_f + wte/ff_out) and train ONLY the WoDD modules (write_head + tape_read,
~ a few M params on a 7B base).

Recipe outline (8-week plan):
  - W1-2: this script, frozen backbone, T_step in {1..4}, sparsity 1e-2,
          entropy 1e-3. Goal: GSM8K baseline match + nontrivial mean_gate.
  - W3-4: matched-FLOPs head-to-head vs MetaState / LATTS / split-LLaDA
          baselines. Primary go/no-go.

Run:
    bash scripts/llada_wodd/run.sh
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers
from transformers.trainer_utils import get_last_checkpoint

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    wodd_init_from: str | None = None


@dataclass
class WoDDArguments:
    # ---- Layer partition ----
    prelude_layers: int = 8
    recurrent_layers: int = 16
    coda_layers: int = 8
    # ---- Tape ----
    tape_dim: int = 1024
    tape_max_writes: int = 16
    tape_n_heads: int = 8
    # NOTE: t_step_eval lives on MDLMWoDDConfig (the TrainingArguments parent);
    # the model-config value is mirrored from training_args.t_step_eval at
    # build time so the trainer and the model agree.
    zero_init_wodd: bool = True


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "nvidia/OpenMathInstruct-2[train:500000,test:5000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Mask the loss on prompt tokens."},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMWoDDConfig):
    output_dir: str = ".models/LLaDA-8B-WoDD/openmath2-v0"
    group_by_length: bool = True
    num_train_epochs: float = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2


def get_resume_checkpoint(output_dir: str) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    resume = get_last_checkpoint(output_dir)
    if resume is not None:
        logger.info("Auto-resuming from checkpoint: %s", resume)
    else:
        logger.info(
            "No existing checkpoint found under %s; starting a fresh run.",
            output_dir,
        )
    return resume


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, WoDDArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        wodd_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.print_args(wodd_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    from dllm.pipelines.llada_wodd import LLaDAWoDDModelLM

    resume_checkpoint = get_resume_checkpoint(training_args.output_dir)

    if resume_checkpoint is not None:
        model = LLaDAWoDDModelLM.from_pretrained(resume_checkpoint)
    elif model_args.wodd_init_from is not None:
        logger.info(
            "Warm-starting model weights from WoDD checkpoint: %s "
            "(fresh optimizer / scheduler / step counter)",
            model_args.wodd_init_from,
        )
        model = LLaDAWoDDModelLM.from_pretrained(model_args.wodd_init_from)
    else:
        wodd_kwargs = dict(
            prelude_layers=wodd_args.prelude_layers,
            recurrent_layers=wodd_args.recurrent_layers,
            coda_layers=wodd_args.coda_layers,
            tape_dim=wodd_args.tape_dim,
            tape_max_writes=wodd_args.tape_max_writes,
            tape_n_heads=wodd_args.tape_n_heads,
            # Mirror trainer's t_step_eval into the model config so the two
            # never drift. (Originally duplicated on WoDDArguments, which
            # conflicted with MDLMWoDDConfig at argparse time.)
            t_step_eval=training_args.t_step_eval,
            zero_init_wodd=wodd_args.zero_init_wodd,
        )
        model = LLaDAWoDDModelLM.from_llada_checkpoint(
            model_args.model_name_or_path, **wodd_kwargs
        )

    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start LLaDA-WoDD SFT...")
    trainer = dllm.core.trainers.MDLMWoDDTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=dllm.utils.NoAttentionMaskWrapper(
            transformers.DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                label_pad_token_id=tokenizer.pad_token_id,
            ),
        ),
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
