"""
LLaDA-Looped SFT (latent-feedback loop).

Retrofits a pretrained LLaDA-8B checkpoint with a looped middle-block
mechanism (see /Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py)
and runs SFT via MDLMLoopedTrainer.

Two variants selectable at the command line:

  V2 (default, headline): --use_latent_feedback True
      Loop applies a RecursiveMAS-style Adapter (pre_ln -> Linear -> GELU
      -> Linear -> residual add -> post_ln) to h_r every iteration.

  V1 (ablation): --use_latent_feedback False
      Pure h-recurrence h_{r+1} = R(h_r). No recursive_link.

Default trainable surface (V2): ONLY recursive_link.*. The entire vanilla
LLaDA backbone (prelude + R + coda + ln_f + wte) is frozen, isolating the
loop's contribution and keeping optimizer state tiny (~33M params).

For V1, the V2 freeze profile leaves nothing to train; pass
`--freeze_recurrent_blocks False --freeze_ln_f False --freeze_wte False`
to recover the V1 surface (R blocks + ln_f + wte trainable). See
scripts/looped_llada/run_v1.sh.

For V0 (vanilla LLaDA) full-param baseline, use
/Users/pixeli/dllm/examples/llada/sft.py on the same data.

Run:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    cd /Users/pixeli/dllm

    # V2 main training (FSDP, 8 GPUs):
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/openmath2-v2

    # V2 T_rec=1 warmup:
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/openmath2-v2-stage1 \
        --t_rec_min 1 --t_rec_max 1 --num_train_epochs 0.5

    # V1 ablation (pure h-recurrence):
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/openmath2-v1 \
        --use_latent_feedback False
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


@dataclass
class LoopArguments:
    # ---- Layer partition ----
    prelude_layers: int = 8
    recurrent_layers: int = 16
    coda_layers: int = 8
    # ---- Variant ----
    # True  -> V2 (latent feedback), False -> V1 (pure h-recurrence)
    use_latent_feedback: bool = True
    # ---- Eval-time T_rec written into config ----
    mu_rec_eval: int = 4


@dataclass
class DataArguments(dllm.utils.DataArguments):
    # OpenMathInstruct-2: 14M math problems with step-by-step solutions ending
    # in \\boxed{answer}. Default slice: 500k train + 5k test for first runs.
    dataset_args: str = "nvidia/OpenMathInstruct-2[train:500000,test:5000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Mask the loss on prompt tokens."},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMLoopedConfig):
    output_dir: str = ".models/LLaDA-8B-Looped/openmath2-v2"
    group_by_length: bool = True
    num_train_epochs: float = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    # MDLMLoopedConfig defaults: t_rec_min=1, t_rec_max=6, loop_lr_mult=10,
    # freeze_{prelude,recurrent_blocks,coda,ln_f,wte}=True (i.e. only
    # recursive_link trains). Override here if desired.


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
        (ModelArguments, LoopArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        loop_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.print_args(loop_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    from dllm.pipelines.llada_looped import LLaDALoopedModelLM

    resume_checkpoint = get_resume_checkpoint(training_args.output_dir)

    if resume_checkpoint is not None:
        model = LLaDALoopedModelLM.from_pretrained(resume_checkpoint)
    else:
        loop_kwargs = dict(
            prelude_layers=loop_args.prelude_layers,
            recurrent_layers=loop_args.recurrent_layers,
            coda_layers=loop_args.coda_layers,
            use_latent_feedback=loop_args.use_latent_feedback,
            mu_rec_eval=loop_args.mu_rec_eval,
        )
        model = LLaDALoopedModelLM.from_llada_checkpoint(
            model_args.model_name_or_path, **loop_kwargs
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
    logger.info("Start LLaDA-Looped SFT...")
    trainer = dllm.core.trainers.MDLMLoopedTrainer(
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
