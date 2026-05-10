"""LLaDA TD distillation entry (Phase 2 / P0).

Trains a vanilla LLaDA student against a cached teacher trajectory using
coarse KL with a tail bucket. No inner loop, no re-entry. P0's only job
is to verify that progressive distillation (teacher 256 -> student 128
diffusion steps) survives at all on discrete diffusion math reasoning,
before any architectural complexity is added.

The student is initialised from the teacher SFT checkpoint and
fine-tuned; the architecture is unchanged.

Run::

    bash scripts/looped_llada/td_p0.sh
"""

import os
from dataclasses import dataclass

import accelerate
import transformers
from transformers.trainer_utils import get_last_checkpoint

import dllm
from dllm.core.trainers.td_distill import (
    TDDistillationTrainer,
    TDDistillCollator,
    TDDistillConfig,
    TrajectoryCacheDataset,
)

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    """Student initialisation. Default: start from the teacher SFT
    checkpoint that was used to build the cache."""

    model_name_or_path: str = ""


def _get_resume_checkpoint(output_dir: str) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    resume = get_last_checkpoint(output_dir)
    if resume is not None:
        logger.info("Auto-resuming from %s", resume)
    return resume


def _build_dataset(
    cache_dir: str,
    student_steps: int,
    teacher_stride: int,
    sample_id_filter,
    allow_stale: bool,
) -> TrajectoryCacheDataset:
    return TrajectoryCacheDataset(
        cache_dir=cache_dir,
        student_steps=student_steps,
        teacher_stride=teacher_stride,
        sample_id_filter=sample_id_filter,
        allow_stale_cache=allow_stale,
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TDDistillConfig))
    model_args, training_args = parser.parse_args_into_dataclasses()

    if not training_args.cache_dir:
        raise ValueError("--cache_dir is required")
    if not model_args.model_name_or_path:
        raise ValueError("--model_name_or_path (teacher init) is required")

    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    resume_checkpoint = _get_resume_checkpoint(training_args.output_dir)
    init_from = resume_checkpoint or model_args.model_name_or_path
    logger.info("Initialising student from %s", init_from)
    model = dllm.utils.get_model(model_args=model_args, model_name_or_path=init_from)

    with accelerate.PartialState().local_main_process_first():
        full_ds = _build_dataset(
            cache_dir=training_args.cache_dir,
            student_steps=training_args.student_steps,
            teacher_stride=training_args.teacher_stride,
            sample_id_filter=None,
            allow_stale=training_args.allow_stale_cache,
        )
        if training_args.val_fraction > 0:
            train_ids, val_ids = TrajectoryCacheDataset.split_sample_ids(
                sample_ids=full_ds.all_sample_ids(),
                val_fraction=training_args.val_fraction,
            )
            logger.info(
                "Train/val split by sample_id: %d train, %d val",
                len(train_ids),
                len(val_ids),
            )
            train_ds = _build_dataset(
                cache_dir=training_args.cache_dir,
                student_steps=training_args.student_steps,
                teacher_stride=training_args.teacher_stride,
                sample_id_filter=train_ids,
                allow_stale=training_args.allow_stale_cache,
            )
            val_ds = _build_dataset(
                cache_dir=training_args.cache_dir,
                student_steps=training_args.student_steps,
                teacher_stride=training_args.teacher_stride,
                sample_id_filter=val_ids,
                allow_stale=training_args.allow_stale_cache,
            )
        else:
            train_ds = full_ds
            val_ds = None

    accelerate.PartialState().wait_for_everyone()
    logger.info(
        "Train dataset size = %d (samples * student_steps); val size = %s",
        len(train_ds),
        len(val_ds) if val_ds is not None else "0",
    )

    collator = TDDistillCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = TDDistillationTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
