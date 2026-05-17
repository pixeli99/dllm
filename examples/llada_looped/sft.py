"""
LLaDA-Looped SFT (V2.1-a, feedback-bridge variant).

Retrofits a pretrained LLaDA-8B checkpoint with a looped middle-block
mechanism (see /Users/pixeli/dllm/dllm/pipelines/llada_looped/models/modeling_llada_looped.py)
and runs SFT via MDLMLoopedTrainer.

V2.1-a key invariants:
  - First-pass bypass: RecursiveLink only applied on feedback transitions
    (r >= 1), so T_rec=1 is exactly vanilla split LLaDA, by construction.
  - Full BPTT: gradient flows end-to-end through all iterations (no
    detach), so every iter's link/R gets direct credit from L_final.

Two variants selectable at the command line:

  V2.1-a (default, headline): --use_latent_feedback True
      h_0 = e (prelude output)
      h_1 = R(h_0)                          # vanilla first pass
      h_r = R(RecursiveLink(h_{r-1}))       # feedback transition (r >= 2)
      # Optional: --use_damped_update True commits feedback transitions via
      # h_r = h_{r-1} + alpha_r * (proposed_h_r - h_{r-1}), where alpha_r
      # can be constant, normalized by T_rec, or decayed across feedback steps.

      Default training T_rec ~ Uniform{2..6} (T=1 has no link gradient).
      Default trainable surface: ONLY recursive_link.* (~33M params).
      The vanilla LLaDA backbone (prelude + R + coda + ln_f + wte) is
      frozen by default to isolate the loop's contribution.

      Memory: full BPTT through T_rec=6 multiplies activation memory ~6x
      over a single forward. If you OOM, add `--gradient_checkpointing
      True` to the run script.

  V1 (ablation): --use_latent_feedback False
      h_0 = e
      h_1 = R(h_0)                          # vanilla first pass
      h_r = R(h_{r-1})                      # pure recurrence (r >= 2)

      For V1, the V2 freeze profile leaves nothing to train; pass
      `--freeze_recurrent_blocks False --freeze_ln_f False --freeze_wte
      False` to recover the V1 surface. See scripts/looped_llada/run_v1.sh.

For V0 (vanilla LLaDA) full-param baseline, use
/Users/pixeli/dllm/examples/llada/sft.py on the same data.

Run:
    # V2.1-a main training (frozen R, only recursive_link trains):
    bash scripts/looped_llada/run.sh

    # V2.1-a + R unfreeze (P2a, larger trainable surface):
    bash scripts/looped_llada/run_unfrozen_r.sh

    # V1 ablation (pure h-recurrence):
    bash scripts/looped_llada/run_v1.sh
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
    # Optional warm-start from an existing looped checkpoint. When set,
    # the trainer loads MODEL WEIGHTS ONLY from this path (not optimizer
    # state) and starts a fresh training run from step 0 -- so the LR
    # scheduler / freeze profile / T_rec range are taken from the current
    # invocation, NOT from the warm-start checkpoint. Used for two-stage
    # training (e.g. P1-frozen-R warmup -> P2a-unfrozen-R fine-tune).
    looped_init_from: str | None = None


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
    # ---- Optional feedback damping ----
    use_damped_update: bool = False
    damping_schedule: str = "constant"
    damping_alpha: float = 0.5
    learn_damping_alpha: bool = False
    damping_tau: float = 1.0
    learn_damping_tau: bool = True
    damping_decay_beta: float = 0.0
    # ---- Diffusion-conditioned damping ----
    # Keeps T_rec=1 unchanged, but for feedback passes lets tau depend on
    # current mask ratio and gates updates on masked vs unmasked tokens.
    use_diffusion_conditioned_damping: bool = False
    damping_masked_gate: float = 1.0
    damping_unmasked_gate: float = 0.1
    learn_mask_tau: bool = True
    mask_tau_ref: float = 0.5
    mask_tau_slope_init: float = 0.0
    # ---- Feedback-only workspace slots ----
    # workspace_size=0 is the baseline. When >0, slots are appended only inside
    # feedback recurrent passes (r >= 1), so T_rec=1 keeps the vanilla path.
    workspace_size: int = 0
    workspace_init_std: float = 0.02


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
        # In-place resume: load model + optimizer state from output_dir.
        model = LLaDALoopedModelLM.from_pretrained(resume_checkpoint)
    elif model_args.looped_init_from is not None:
        # Warm-start: load model weights from a separate looped checkpoint,
        # but DO NOT pass resume_from_checkpoint to trainer.train() so the
        # optimizer / LR scheduler / step counter all start fresh. Used to
        # chain stages (e.g. frozen-R warmup followed by R unfreeze).
        logger.info(
            "Warm-starting model weights from looped checkpoint: %s "
            "(fresh optimizer / scheduler / step counter)",
            model_args.looped_init_from,
        )
        model = LLaDALoopedModelLM.from_pretrained(model_args.looped_init_from)
    else:
        loop_kwargs = dict(
            prelude_layers=loop_args.prelude_layers,
            recurrent_layers=loop_args.recurrent_layers,
            coda_layers=loop_args.coda_layers,
            use_latent_feedback=loop_args.use_latent_feedback,
            mu_rec_eval=loop_args.mu_rec_eval,
            use_damped_update=loop_args.use_damped_update,
            damping_schedule=loop_args.damping_schedule,
            damping_alpha=loop_args.damping_alpha,
            learn_damping_alpha=loop_args.learn_damping_alpha,
            damping_tau=loop_args.damping_tau,
            learn_damping_tau=loop_args.learn_damping_tau,
            damping_decay_beta=loop_args.damping_decay_beta,
            use_diffusion_conditioned_damping=(
                loop_args.use_diffusion_conditioned_damping
            ),
            damping_masked_gate=loop_args.damping_masked_gate,
            damping_unmasked_gate=loop_args.damping_unmasked_gate,
            learn_mask_tau=loop_args.learn_mask_tau,
            mask_tau_ref=loop_args.mask_tau_ref,
            mask_tau_slope_init=loop_args.mask_tau_slope_init,
            workspace_size=loop_args.workspace_size,
            workspace_init_std=loop_args.workspace_init_std,
        )
        model = LLaDALoopedModelLM.from_llada_checkpoint(
            model_args.model_name_or_path, **loop_kwargs
        )

    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    if getattr(tokenizer, "mask_token_id", None) is not None:
        model.config.mask_token_id = int(tokenizer.mask_token_id)
        if hasattr(model, "model"):
            model.model.config.mask_token_id = int(tokenizer.mask_token_id)

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
