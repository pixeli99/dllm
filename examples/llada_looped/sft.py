"""
LLaDA-Looped SFT (belief-bottleneck loop).

Retrofits a pretrained LLaDA-8B checkpoint with a looped middle-block
mechanism (see /Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py)
and runs full-parameter SFT via MDLMLoopedTrainer.

Two variants selectable at the command line:

  V2 (default, headline): --use_belief_feedback True
      Loop applies soft-belief readout -> embedding -> gated injection
      every iteration. Drop-in at T_rec=1, alpha=0.

  V1 (ablation): --use_belief_feedback False
      Pure h-recurrence h_{r+1} = R(h_r). No alpha/ln_out/belief_norm.

Trainable surface (default): blocks[prelude_layers : prelude_layers+recurrent_layers]
+ wte (= tied lm_head) + ln_f + (V2 only) alpha/ln_out/belief_norm. Prelude
and coda blocks are frozen by default -- override with `--freeze_prelude False`
or `--freeze_coda False` if you want full-param finetune.

For V0 (vanilla LLaDA, no loop module) full-param baseline, use
/Users/pixeli/dllm/examples/llada/sft.py on the same data. For a
surface-matched V0 baseline (= V1 with T_rec=1), run this entry with
--use_belief_feedback False --t_rec_min 1 --t_rec_max 1.

Run:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    cd /Users/pixeli/dllm

    # V2 main training (FSDP, 8 GPUs):
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/openmath2-v2

    # V2 Stage 1 warmup (alpha frozen, T_rec=1, probe-train ln_out + base):
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/openmath2-v2-stage1 \
        --stage1_freeze_alpha True --num_train_epochs 0.5

    # V1 ablation (pure h-recurrence):
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/openmath2-v1 \
        --use_belief_feedback False
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
    # True  -> V2 (belief bottleneck), False -> V1 (pure h-recurrence)
    use_belief_feedback: bool = True
    # ---- Belief-feedback hyperparameters (V2 only) ----
    alpha_init: float = 0.0
    tau: float = 1.0
    belief_top_k: int = 32
    use_belief_norm: bool = True
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
    # MDLMLoopedConfig defaults: t_rec_min=1, t_rec_max=6, traj_beta=0,
    # loop_lr_mult=10. Override here if desired.


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

    # Model: retrofit vanilla LLaDA-8B into looped variant.
    from dllm.pipelines.llada_looped import LLaDALoopedModelLM

    resume_checkpoint = get_resume_checkpoint(training_args.output_dir)

    if resume_checkpoint is not None:
        # Resuming a looped checkpoint: standard from_pretrained.
        model = LLaDALoopedModelLM.from_pretrained(resume_checkpoint)
    else:
        loop_kwargs = dict(
            prelude_layers=loop_args.prelude_layers,
            recurrent_layers=loop_args.recurrent_layers,
            coda_layers=loop_args.coda_layers,
            use_belief_feedback=loop_args.use_belief_feedback,
            alpha_init=loop_args.alpha_init,
            tau=loop_args.tau,
            belief_top_k=loop_args.belief_top_k,
            use_belief_norm=loop_args.use_belief_norm,
            mu_rec_eval=loop_args.mu_rec_eval,
        )
        model = LLaDALoopedModelLM.from_llada_checkpoint(
            model_args.model_name_or_path, **loop_kwargs
        )

    # Tokenizer.
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Dataset.
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
