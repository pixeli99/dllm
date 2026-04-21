"""
LLaDA-Looped SFT.

Retrofits a pretrained LLaDA-8B checkpoint with a Parcae-style middle-loop
(see /Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py)
and runs full-parameter SFT via MDLMLoopedTrainer.

Run:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    cd /Users/pixeli/dllm

    # Fixed μ_rec = 2, full-param SFT (FSDP, 8 GPUs):
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/tulu-3-mu2-fixed \
        --mu_rec_train_mean 2 --mu_rec_train_min 2 --mu_rec_train_max 2 \
        --recurrence_dist fixed --mu_rec_eval 2

    # Poisson μ_rec, mean=2, range [1, 4]:
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/tulu-3-mu2-poisson

    # t-adaptive schedule + cycle-consistency ablation:
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/tulu-3-t_adapt-cc \
        --t_adaptive True --t_bump_center 0.5 --t_bump_width 0.25 \
        --cycle_consistency_weight 0.1

    # Stage-1 warm-start (freeze base, train only loop controller):
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada_looped/sft.py \
        --output_dir .models/LLaDA-8B-Looped/tulu-3-stage1 \
        --freeze_base True --num_train_epochs 0.5 --learning_rate 1e-3
    # Then resume with --freeze_base False for stage-2 full unfreeze.

    # Baseline for isolation (vanilla LLaDA + same data, no loop):
    accelerate launch \
        --config_file /Users/pixeli/dllm/scripts/accelerate_configs/fsdp.yaml \
        /Users/pixeli/dllm/examples/llada/sft.py \
        --output_dir .models/LLaDA-8B-Base/tulu-3-baseline
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class LoopArguments:
    prelude_layers: int = 8
    recurrent_layers: int = 16
    coda_layers: int = 8
    mu_rec_eval: int = 2
    use_input_norm: bool = True
    use_diag_B: bool = False
    A_init_log: float = -4.0
    delta_init: float = 1.0
    B_init_scale: float = 0.0


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Mask the loss on prompt tokens."},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMLoopedConfig):
    output_dir: str = ".models/LLaDA-8B-Looped/tulu-3-mu2"
    group_by_length: bool = True
    num_train_epochs: float = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    # Looping defaults (see MDLMLoopedConfig for full list).
    mu_rec_train_mean: float = 2.0
    mu_rec_train_min: int = 1
    mu_rec_train_max: int = 4
    recurrence_dist: str = "poisson"
    mu_bwd_ratio: float = 0.5


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
    dllm.utils.print_args_main(model_args, loop_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # Model: retrofit LLaDA-8B into a looped variant with drop-in weight load.
    from dllm.pipelines.llada_looped import LLaDALoopedModelLM

    loop_kwargs = dict(
        prelude_layers=loop_args.prelude_layers,
        recurrent_layers=loop_args.recurrent_layers,
        coda_layers=loop_args.coda_layers,
        mu_rec_eval=loop_args.mu_rec_eval,
        use_input_norm=loop_args.use_input_norm,
        use_diag_B=loop_args.use_diag_B,
        A_init_log=loop_args.A_init_log,
        delta_init=loop_args.delta_init,
        B_init_scale=loop_args.B_init_scale,
    )
    model = LLaDALoopedModelLM.from_llada_checkpoint(
        model_args.model_name_or_path, **loop_kwargs
    )

    # Tokenizer: LLaDALoopedModelLM is a subclass of LLaDAModelLM, so
    # get_tokenizer applies the LLaDA branch and installs the mdm mask token.
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Dataset
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
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
