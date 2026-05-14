"""Phase 2 unified SFT entry: one script for all five arms.

WODD_PLAN.md §3.2 matched-FLOPs head-to-head. Same OpenMath subset, same
optimizer, same training tokens, 5 seeds. We dispatch the model + freezing
based on the --arm flag and then defer to either the vanilla MDLMTrainer
(A0/A1/A2/A3) or MDLMWoDDTrainer (A4) so the train loop is identical
across arms except for the bits each arm actually requires.

Arms (from WODD_PLAN.md):
  vanilla   A0  : no adapter; trainable=0; reference forward.
  latts     A1  : LATTS (latent test-time steps); trainable=0; T_step=4.
  metastate A2  : persistent state buffer; ~0.6 % trainable; T_step=4.
  dpad      A3  : suffix scratchpad; ~1 % trainable; T_step=1.
  wodd      A4  : append-only tape; ~1 % trainable; T_step=4.

Phase 2 launch (8 × H800):
    accelerate launch \\
      --config_file scripts/accelerate_configs/fsdp.yaml \\
      examples/phase2_sft.py \\
      --arm metastate --seed 0 \\
      --output_dir .models/phase2/metastate/seed0 \\
      --dataset_args ".data/sft/llada/openmath2-500k[train:500000,test:5000]" \\
      --load_preprocessed_data True \\
      ... (HF TrainingArguments) ...

Per WODD_PLAN.md §3.2 the head-to-head requires 5 seeds × 5 arms.
Loop over arms × seeds with `scripts/phase2/launch_all.sh` (or via the
SLURM array launcher).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import accelerate
import transformers
from transformers.trainer_utils import get_last_checkpoint

import dllm

logger = dllm.utils.get_default_logger(__name__)


# ---------------------------------------------------------------------------
# Per-arm dispatch table
# ---------------------------------------------------------------------------

def _build_vanilla(model_name, dtype, _arm_kwargs):
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    return LLaDAModelLM.from_pretrained(model_name, torch_dtype=dtype)


def _build_latts(model_name, dtype, arm_kwargs):
    from dllm.pipelines.llada_latts import LLaDALATTSModelLM
    return LLaDALATTSModelLM.from_llada_checkpoint(
        model_name, torch_dtype=dtype, **arm_kwargs,
    )


def _build_metastate(model_name, dtype, arm_kwargs):
    from dllm.pipelines.llada_metastate import LLaDAMetaStateModelLM
    return LLaDAMetaStateModelLM.from_llada_checkpoint(
        model_name, torch_dtype=dtype, **arm_kwargs,
    )


def _build_dpad(model_name, dtype, arm_kwargs):
    from dllm.pipelines.llada_dpad import LLaDADPadModelLM
    return LLaDADPadModelLM.from_llada_checkpoint(
        model_name, torch_dtype=dtype, **arm_kwargs,
    )


def _build_wodd(model_name, dtype, arm_kwargs):
    from dllm.pipelines.llada_wodd import LLaDAWoDDModelLM
    return LLaDAWoDDModelLM.from_llada_checkpoint(
        model_name, torch_dtype=dtype, **arm_kwargs,
    )


ARM_BUILDERS = {
    "vanilla":   _build_vanilla,
    "latts":     _build_latts,
    "metastate": _build_metastate,
    "dpad":      _build_dpad,
    "wodd":      _build_wodd,
}


# Prefixes of param names that the arm considers "trainable adapter". All
# other params (the LLaDA backbone) get requires_grad_(False).
ARM_TRAINABLE_PREFIXES = {
    "vanilla":   (),                                    # nothing extra
    "latts":     (),                                    # nothing extra
    "metastate": ("model.state_init.", "model.state_read.", "model.state_update."),
    "dpad":      ("model.scratchpad",),                 # single tensor
    "wodd":      ("model.write_head.", "model.tape_read.", "model.write_pool."),
}


# Default per-arm config kwargs (handed to from_llada_checkpoint). The
# layer partition matches across arms; per-arm capacity is set here.
DEFAULT_ARM_KWARGS = {
    "vanilla":   {},
    "latts":     dict(prelude_layers=8, recurrent_layers=16, coda_layers=8,
                      t_step_eval=4),
    "metastate": dict(prelude_layers=8, recurrent_layers=16, coda_layers=8,
                      t_step_eval=4,
                      state_slots=64, state_dim=1024, state_n_heads=8,
                      zero_init_metastate=True),
    "dpad":      dict(prelude_layers=8, recurrent_layers=16, coda_layers=8,
                      dpad_len=512, zero_init_dpad=True),
    "wodd":      dict(prelude_layers=8, recurrent_layers=16, coda_layers=8,
                      t_step_eval=4, tape_dim=1024,
                      # multi-slot: K*S = 4*16 = 64 tape entries per forward
                      tape_max_writes=64,
                      write_slots=16, write_pool_n_heads=8,
                      tape_n_heads=8, zero_init_wodd=True),
}


# ---------------------------------------------------------------------------
# CLI dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArmArguments:
    arm: str = field(
        default="wodd",
        metadata={"choices": list(ARM_BUILDERS.keys()),
                  "help": "Which Phase 2 arm to train."}
    )
    arm_kwargs_json: str = field(
        default="{}",
        metadata={"help": "JSON dict of extra arm config kwargs that "
                          "override DEFAULT_ARM_KWARGS for this arm."}
    )


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "/lustre/projects/polyullm/lipengxiang_tmp/LLaDA-8B-Instruct"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "nvidia/OpenMathInstruct-2[train:500000,test:5000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Mask the loss on prompt tokens."},
    )


# We use the WoDD trainer's config dataclass as the superset; for non-WoDD
# arms the WoDD-specific fields (write_sparsity_lambda etc) are simply
# ignored because we route through MDLMTrainer instead.
@dataclass
class TrainingArguments(dllm.core.trainers.MDLMWoDDConfig):
    output_dir: str = ".models/phase2/wodd/seed0"
    group_by_length: bool = True
    num_train_epochs: float = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2


# ---------------------------------------------------------------------------
# Freezing helper
# ---------------------------------------------------------------------------

def freeze_to_prefixes(model, allowed_prefixes):
    """Set requires_grad=False on every param whose name does NOT start with
    one of `allowed_prefixes`. Returns (n_train, n_total)."""
    if not allowed_prefixes:
        # Nothing to train: leave the model as-is. The training script will
        # warn the user.
        n = sum(p.numel() for p in model.parameters())
        return n, n
    n_train = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if any(name == pref or name.startswith(pref) for pref in allowed_prefixes):
            p.requires_grad_(True)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
    return n_train, n_total


# ---------------------------------------------------------------------------
# Train entry
# ---------------------------------------------------------------------------

def train():
    import json

    parser = transformers.HfArgumentParser(
        (ArmArguments, ModelArguments, DataArguments, TrainingArguments)
    )
    arm_args, model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses()
    )
    arm = arm_args.arm.lower()
    assert arm in ARM_BUILDERS, f"unknown --arm {arm}"

    dllm.utils.print_args_main(model_args, data_args, training_args)
    logger.info(f"[phase2] arm={arm}")
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # Resolve arm kwargs: defaults overlaid with user JSON.
    arm_kwargs = dict(DEFAULT_ARM_KWARGS[arm])
    user_kwargs = json.loads(arm_args.arm_kwargs_json or "{}")
    arm_kwargs.update(user_kwargs)
    if arm_kwargs:
        logger.info("[phase2] arm_kwargs: %s", arm_kwargs)

    # Auto-resume.
    resume_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)
        if resume_checkpoint is not None:
            logger.info("[phase2] auto-resuming from %s", resume_checkpoint)
            # When resuming we let from_pretrained load the saved config so we
            # don't override with the default kwargs.
            ARM_MODEL_LM = {
                "vanilla": "dllm.pipelines.llada.models.modeling_llada:LLaDAModelLM",
                "latts":   "dllm.pipelines.llada_latts:LLaDALATTSModelLM",
                "metastate": "dllm.pipelines.llada_metastate:LLaDAMetaStateModelLM",
                "dpad":    "dllm.pipelines.llada_dpad:LLaDADPadModelLM",
                "wodd":    "dllm.pipelines.llada_wodd:LLaDAWoDDModelLM",
            }
            modpath, clsname = ARM_MODEL_LM[arm].split(":")
            import importlib
            cls = getattr(importlib.import_module(modpath), clsname)
            model = cls.from_pretrained(resume_checkpoint)
        else:
            logger.info("[phase2] no checkpoint in %s; fresh run", training_args.output_dir)

    if resume_checkpoint is None:
        # bf16 by default; mirrors run.sh.
        import torch
        dtype = torch.bfloat16 if training_args.bf16 else (
            torch.float16 if training_args.fp16 else torch.float32
        )
        model = ARM_BUILDERS[arm](model_args.model_name_or_path, dtype, arm_kwargs)

    # Apply freezing.
    n_train, n_total = freeze_to_prefixes(model, ARM_TRAINABLE_PREFIXES[arm])
    pct = 100.0 * n_train / max(1, n_total)
    logger.info(
        "[phase2 freeze] arm=%s trainable=%d / total=%d (%.4f %%)",
        arm, n_train, n_total, pct,
    )
    if arm == "vanilla":
        logger.warning(
            "[phase2] arm=vanilla has 0 trainable adapter params -- this run "
            "is only useful if the user explicitly intends to fine-tune the "
            "backbone (set ARM_TRAINABLE_PREFIXES[vanilla] accordingly) or as "
            "an inference-only run."
        )

    # Dataset
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
                map_fn, num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)
    accelerate.PartialState().wait_for_everyone()

    # Trainer dispatch: WoDD has its own three-component loss; the rest
    # use plain MDLMTrainer.
    if arm == "wodd":
        TrainerCls = dllm.core.trainers.MDLMWoDDTrainer
    else:
        TrainerCls = dllm.core.trainers.MDLMTrainer

    logger.info("[phase2] launching %s on arm=%s", TrainerCls.__name__, arm)
    trainer = TrainerCls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=dllm.utils.NoAttentionMaskWrapper(
            transformers.DataCollatorForSeq2Seq(
                tokenizer, return_tensors="pt", padding=True,
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
