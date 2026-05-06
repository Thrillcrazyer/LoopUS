"""Shared training runtime for DDD entrypoints."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, get_cosine_schedule_with_warmup

from models.configuration_lds import LDSConfig
from models.modeling_lds import LDSForCausalLM
from utils.common import TrainConfig, set_seed
from utils.ddp_helper import _unwrap_ddp
from utils.lm_eval_harness import DEFAULT_TASKS, format_results_log, run_lm_eval_for_training
from utils.metrics import (
    BinaryClassificationCounts,
    MetricsTracker,
    append_jsonl,
    build_eval_response,
    compute_q_target,
    summarize_eval_step,
    summarize_q_metric_counts,
    update_q_metric_counts,
)


DataLoaderFactory = Callable[
    [TrainConfig, Any, Accelerator],
    tuple[DataLoader, DataLoader | None, int, int],
]


def _uses_sharded_parameters(accelerator: Accelerator) -> bool:
    """Return whether parameter fetches require cross-rank participation."""
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true":
        return True

    state = getattr(accelerator, "state", None)
    if getattr(state, "deepspeed_plugin", None) is not None:
        return True
    if getattr(state, "fsdp_plugin", None) is not None:
        return True

    distributed_type = getattr(accelerator.state, "distributed_type", None)
    if distributed_type is None:
        distributed_type = getattr(accelerator, "distributed_type", None)
    distributed_name = getattr(distributed_type, "name", str(distributed_type)).upper()
    return distributed_name in {"DEEPSPEED", "FSDP"}


@dataclass
class TrainingSetup:
    """State created before entering the training loop."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR | ApproximateTokenCosineScheduler | None
    dataloader: DataLoader
    eval_dataloader: DataLoader | None
    steps_per_epoch: int = 0
    global_step: int = 0
    start_epoch: int = 0
    start_step: int = 0


class ApproximateTokenCosineScheduler:
    """Cosine scheduler driven by an approximate consumed-token count."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_tokens: int,
        warmup_tokens: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.total_tokens = max(total_tokens, 1)
        self.warmup_tokens = max(warmup_tokens, 0)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_lrs = list(self.base_lrs)
        self.tokens_seen = 0

    def _lr_scale(self) -> float:
        if self.tokens_seen <= 0:
            return 0.0 if self.warmup_tokens > 0 else 1.0

        if self.warmup_tokens > 0 and self.tokens_seen < self.warmup_tokens:
            return self.tokens_seen / max(self.warmup_tokens, 1)

        if self.total_tokens <= self.warmup_tokens:
            return 1.0

        progress = (self.tokens_seen - self.warmup_tokens) / max(
            self.total_tokens - self.warmup_tokens,
            1,
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    def step(self, tokens: int) -> None:
        """Advance the scheduler by an approximate token count."""
        self.tokens_seen = min(self.total_tokens, self.tokens_seen + max(tokens, 0))
        scale = self._lr_scale()
        self.last_lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            new_lr = base_lr * scale
            group["lr"] = new_lr
            self.last_lrs.append(new_lr)

    def get_last_lr(self) -> list[float]:
        """Return the current learning rates in optimizer param-group order."""
        return self.last_lrs


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
    accelerator: Accelerator,
    train_samples: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR | ApproximateTokenCosineScheduler | None:
    """Build the configured scheduler from step or approximate token budgets."""
    if cfg.lr_scheduler == "none" or steps_per_epoch <= 0:
        return None

    if cfg.lr_scheduler == "cosine":
        num_training_steps = steps_per_epoch * cfg.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=num_training_steps,
        )
        if accelerator.is_main_process:
            print(
                f"[scheduler] Cosine with warmup: {cfg.warmup_steps} warmup / "
                f"{num_training_steps} total steps "
                f"({steps_per_epoch} batches x {cfg.epochs} epochs)"
            )
        return scheduler

    tokens_per_supervision = cfg.batch_size * accelerator.num_processes * cfg.max_length
    tokens_per_update = tokens_per_supervision * cfg.gradient_accumulation_steps
    if train_samples is not None and train_samples > 0:
        total_tokens = train_samples * cfg.max_length * cfg.epochs * cfg.n_supervision
    else:
        total_tokens = steps_per_epoch * cfg.epochs * cfg.n_supervision * tokens_per_supervision
    warmup_tokens = cfg.warmup_steps * tokens_per_update
    scheduler = ApproximateTokenCosineScheduler(
        optimizer=optimizer,
        total_tokens=total_tokens,
        warmup_tokens=warmup_tokens,
    )
    if accelerator.is_main_process:
        print(
            f"[scheduler] Approx cosine over tokens: {warmup_tokens} warmup / "
            f"{total_tokens} total tokens "
            f"(~{tokens_per_supervision} tokens per supervision loss, "
            f"~{tokens_per_update} tokens per optimizer update, "
            f"n_supervision={cfg.n_supervision}, "
            f"max_reasoning_steps={cfg.n_reasoning_steps}, "
            f"grad_accum={cfg.gradient_accumulation_steps})"
        )
    return scheduler


def _is_token_scheduler(
    scheduler: torch.optim.lr_scheduler.LambdaLR | ApproximateTokenCosineScheduler | None,
) -> bool:
    """Return whether the configured scheduler advances on token counts."""
    return isinstance(scheduler, ApproximateTokenCosineScheduler)


def _approx_tokens_per_batch(
    input_ids: torch.Tensor,
    cfg: TrainConfig,
    accelerator: Accelerator,
) -> int:
    """Approximate tokens contributing to one supervised loss across all ranks."""
    return input_ids.size(0) * cfg.max_length * accelerator.num_processes


def _estimate_optimizer_updates(step_count: int, cfg: TrainConfig) -> int:
    """Estimate optimizer updates from supervised losses and grad accumulation."""
    if step_count <= 0:
        return 0
    total_supervised_losses = step_count * cfg.n_supervision
    grad_accum = max(cfg.gradient_accumulation_steps, 1)
    return (total_supervised_losses + grad_accum - 1) // grad_accum


def configure_external_services() -> None:
    """Load dotenv credentials and authenticate optional external services."""
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)

    wandb_key = os.getenv("WANDB_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)


def create_accelerator(cfg: TrainConfig) -> Accelerator:
    """Create the shared accelerator configuration."""
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator_kwargs: dict[str, Any] = {
        "kwargs_handlers": [init_kwargs],
        "log_with": "wandb" if cfg.wandb_enabled else None,
    }

    using_deepspeed_config_file = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        and os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", "none") != "none"
    )
    if not using_deepspeed_config_file:
        accelerator_kwargs["mixed_precision"] = cfg.mixed_precision
        accelerator_kwargs["gradient_accumulation_steps"] = cfg.gradient_accumulation_steps

    return Accelerator(**accelerator_kwargs)


def initialize_wandb(cfg: TrainConfig, accelerator: Accelerator) -> None:
    """Initialize W&B only on the main process."""
    if not (cfg.wandb_enabled and accelerator.is_main_process):
        return

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name or None,
        entity=cfg.wandb_entity or None,
        config={k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        resume="allow",
    )


def finalize_wandb(cfg: TrainConfig, accelerator: Accelerator) -> None:
    """Close the W&B run on the main process."""
    if cfg.wandb_enabled and accelerator.is_main_process:
        wandb.finish()


def save_final_model(
    combined_model: LDSForCausalLM,
    cfg: TrainConfig,
    accelerator: Accelerator,
) -> None:
    """Persist the final model weights for the completed run."""
    state_dict = accelerator.get_state_dict(combined_model)
    if not accelerator.is_main_process:
        return

    final_dir = os.path.join(cfg.checkpoint_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(final_dir, "combined_model.pt"))
    print(f"[checkpoint] Final model saved to {final_dir}")


def push_final_model_to_hub(
    combined_model: LDSForCausalLM,
    cfg: TrainConfig,
    accelerator: Accelerator,
) -> None:
    """Push the final training state to HuggingFace Hub once training completes."""
    if not accelerator.is_main_process or not cfg.hub_repo_id:
        return

    print(f"[final] Pushing final model to HuggingFace Hub: {cfg.hub_repo_id}")
    combined_model.push_to_hub(
        cfg.hub_repo_id,
        commit_message="final model",
        private=cfg.hub_private,
    )
    print("[final] Final model push complete.")


def prepare_model(
    model_name: str,
    encoder_layers: list[int] | None = None,
    decoder_layers: list[int] | None = None,
    n_reasoning_steps: int = 100,
    from_hub: str = "",
) -> LDSForCausalLM:
    """Load a pretrained model and decompose it into encoder / reasoning / decoder."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if from_hub:
        print(f"[prepare_model] Loading LDS model from: {from_hub}")
        return LDSForCausalLM.from_pretrained(
            from_hub,
            torch_dtype=dtype,
        )

    base_config_dict = AutoConfig.from_pretrained(model_name).to_dict()
    print(base_config_dict)
    config = LDSConfig(
        base_model_name_or_path=model_name,
        base_config_dict=base_config_dict,
        encoder_layer_indices=encoder_layers,
        decoder_layer_indices=decoder_layers,
        N=n_reasoning_steps,
    )

    return LDSForCausalLM.from_pretrained(
        config,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )


def resolve_sample_budget(cfg: TrainConfig, accelerator: Accelerator) -> int:
    """Resolve sample count from explicit sample or token budgets."""
    max_samples = cfg.train_max_samples
    if cfg.train_max_tokens > 0:
        max_samples = cfg.train_max_tokens // cfg.max_length
        if accelerator.is_main_process:
            print(
                f"[data] Token budget {cfg.train_max_tokens:,} / max_length {cfg.max_length} "
                f"= {max_samples:,} samples"
            )
    return max_samples


def estimate_steps_per_epoch(n_train: int, batch_size: int, num_processes: int) -> int | None:
    """Estimate iterable-dataloader steps per epoch from sample count."""
    if n_train <= 0:
        return None
    return n_train // max(batch_size * num_processes, 1)


def log_dataset_overview(
    cfg: TrainConfig,
    accelerator: Accelerator,
    n_train: int,
    n_eval: int,
    prefix: str,
) -> None:
    """Print the selected dataset split and estimated sizes."""
    if not accelerator.is_main_process:
        return

    config_str = f"/{cfg.train_config}" if cfg.train_config else ""
    print(
        f"[{prefix}] {cfg.train_dataset}{config_str}:{cfg.train_split} | "
        f"train ~{n_train} | eval ~{n_eval}"
    )


def log_lm_eval_plan(cfg: TrainConfig, accelerator: Accelerator) -> None:
    """Log lm-eval settings before training starts."""
    if not (cfg.lm_eval_at_save and accelerator.is_main_process):
        return

    lm_eval_task_list = (
        [task.strip() for task in cfg.lm_eval_tasks.split(",") if task.strip()]
        if cfg.lm_eval_tasks else list(DEFAULT_TASKS)
    )
    print(f"[lm-eval] Will evaluate at each checkpoint save on: {lm_eval_task_list}")
    print(f"[lm-eval] limit={cfg.lm_eval_limit}, batch_size={cfg.lm_eval_batch_size}")


def _setup_training(
    combined_model: LDSForCausalLM,
    accelerator: Accelerator,
    dataloader: DataLoader,
    cfg: TrainConfig,
    eval_dataloader: DataLoader | None = None,
    estimated_steps_per_epoch: int | None = None,
    estimated_train_samples: int | None = None,
) -> TrainingSetup:
    """Create optimizer/scheduler state and prepare distributed components."""
    combined_model.train()
    combined_model.gradient_checkpointing = True

    optimizer = torch.optim.AdamW(list(combined_model.parameters()), lr=cfg.learning_rate)

    global_step = 0
    start_epoch = 0
    start_step = 0

    if cfg.resume:
        resolved = LDSForCausalLM.resolve_resume_path(cfg.resume, cfg.checkpoint_dir)
        if os.path.isdir(resolved):
            training_state = combined_model.load_checkpoint(resolved, optimizer)
            global_step = training_state["global_step"]
            start_epoch = training_state["epoch"]
            if accelerator.is_main_process:
                print(f"[resume] Resuming from step {global_step}, epoch {start_epoch}")
        elif accelerator.is_main_process:
            print(f"[resume] Checkpoint not found at {resolved}, starting from scratch")

    combined_model, optimizer, dataloader = accelerator.prepare(
        combined_model, optimizer, dataloader
    )

    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    try:
        steps_per_epoch = len(dataloader)
    except TypeError:
        steps_per_epoch = estimated_steps_per_epoch or 0

    scheduler = _build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        steps_per_epoch=steps_per_epoch,
        accelerator=accelerator,
        train_samples=estimated_train_samples,
    )

    if cfg.resume and global_step > 0:
        if cfg.start_step >= 0:
            start_step = cfg.start_step
        elif steps_per_epoch > 0:
            start_step = global_step % steps_per_epoch

        if scheduler is not None:
            if _is_token_scheduler(scheduler):
                resume_tokens = (
                    global_step
                    * cfg.n_supervision
                    * cfg.batch_size
                    * accelerator.num_processes
                    * cfg.max_length
                )
                scheduler.step(resume_tokens)
            else:
                for _ in range(global_step):
                    cast(torch.optim.lr_scheduler.LambdaLR, scheduler).step()

        if accelerator.is_main_process:
            suffix = " (overridden by --start-step)" if cfg.start_step >= 0 else ""
            print(
                f"[resume] steps_per_epoch={steps_per_epoch}, "
                f"start_step={start_step} (will skip {start_step} batches){suffix}"
            )

    return TrainingSetup(
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        steps_per_epoch=steps_per_epoch,
        global_step=global_step,
        start_epoch=start_epoch,
        start_step=start_step,
    )


def _compute_supervision_loss(
    combined_model: LDSForCausalLM,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
    plateau_kwargs: dict,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a supervised reasoning step and return all tracked loss components."""
    hidden_states_old = hidden_states.detach()
    hidden_states, q_logit = combined_model.run_final_with_q(hidden_states_old, **plateau_kwargs)
    q_hat = torch.sigmoid(q_logit)

    logits = combined_model.decoder(hidden_states=hidden_states, **plateau_kwargs)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    with torch.no_grad():
        logits_old = combined_model.decoder(hidden_states=hidden_states_old, **plateau_kwargs)
        shift_logits_old = logits_old[:, :-1, :].contiguous()
        lm_loss_prev = F.cross_entropy(
            shift_logits_old.view(-1, shift_logits_old.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    loss_mono = torch.nn.SiLU()(lm_loss - lm_loss_prev)
    
    y_hat = shift_logits.argmax(dim=-1)
    target_q, token_acc = compute_q_target(y_hat=y_hat, y_true=shift_labels)
    target_q = target_q.to(dtype=q_logit.dtype, device=q_logit.device)
    q_loss = F.binary_cross_entropy_with_logits(q_logit, target_q)

    loss = lm_loss + cfg.beta * loss_mono + q_loss
    return loss, token_acc, q_hat, hidden_states, lm_loss, q_loss


def _save_eval_responses(
    responses: list[dict],
    output_dir: str,
    global_step: int,
    tag: str = "eval",
) -> None:
    """Save decoded evaluation responses to a JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{tag}_step_{global_step}.jsonl")
    with open(filepath, "w", encoding="utf-8") as handle:
        for response in responses:
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")

    print(f"[{tag}][step {global_step}] Saved {len(responses)} responses to {filepath}")


def _run_and_log_evaluation(
    combined_model: LDSForCausalLM,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    cfg: TrainConfig,
    global_step: int,
    epoch: int,
) -> None:
    """Run validation evaluation and dispatch its outputs to configured backends."""
    eval_metrics = evaluate(
        combined_model=combined_model,
        dataloader=eval_dataloader,
        accelerator=accelerator,
        n_supervision=cfg.n_supervision,
        max_responses=cfg.eval_max_responses,
        q_threshold=cfg.q_stop_threshold,
    )

    if not accelerator.is_main_process:
        return

    if cfg.eval_jsonl_path:
        append_jsonl(
            cfg.eval_jsonl_path,
            {
                "global_step": global_step,
                "epoch": epoch + 1,
                "loss": eval_metrics["loss"],
                "lm_loss": eval_metrics["lm_loss"],
                "q_loss": eval_metrics["q_loss"],
                "token_acc": eval_metrics["token_acc"],
                "q_hat_mean": eval_metrics["q_hat_mean"],
                "q_f1": eval_metrics["q_f1"],
                "q_precision": eval_metrics["q_precision"],
                "q_recall": eval_metrics["q_recall"],
                "per_i": eval_metrics["per_i"],
                "per_i_q_confidence": [
                    {"i": step_metrics["i"], "q_confidence": step_metrics["q_hat"]}
                    for step_metrics in eval_metrics["per_i"]
                ],
            },
        )

    if cfg.wandb_enabled:
        wandb.log(
            {
                "eval/loss": eval_metrics["loss"],
                "eval/lm_loss": eval_metrics["lm_loss"],
                "eval/q_loss": eval_metrics["q_loss"],
                "eval/token_acc": eval_metrics["token_acc"],
                "eval/q_hat_mean": eval_metrics["q_hat_mean"],
                "eval/q_f1": eval_metrics["q_f1"],
                "eval/q_precision": eval_metrics["q_precision"],
                "eval/q_recall": eval_metrics["q_recall"],
            },
            step=global_step,
        )

    if cfg.eval_responses_dir and eval_metrics.get("responses"):
        _save_eval_responses(
            responses=eval_metrics["responses"],
            output_dir=cfg.eval_responses_dir,
            global_step=global_step,
            tag="eval",
        )

    per_i_acc = " | ".join(
        f"i={step_metrics['i']}:acc={step_metrics['token_acc']:.4f}"
        for step_metrics in eval_metrics["per_i"]
    )
    per_i_q = " | ".join(
        f"i={step_metrics['i']}:q={step_metrics['q_hat']:.4f}"
        for step_metrics in eval_metrics["per_i"]
    )
    per_i_qf1 = " | ".join(
        f"i={step_metrics['i']}:F1={step_metrics['q_f1']:.4f}"
        f"(P={step_metrics['q_precision']:.4f}/R={step_metrics['q_recall']:.4f})"
        for step_metrics in eval_metrics["per_i"]
    )
    print(
        f"[eval][step {global_step}] "
        f"loss={eval_metrics['loss']:.4f} "
        f"lm={eval_metrics['lm_loss']:.4f} "
        f"q={eval_metrics['q_loss']:.4f} "
        f"token_acc={eval_metrics['token_acc']:.4f} "
        f"q_hat={eval_metrics['q_hat_mean']:.4f} "
        f"q_f1={eval_metrics['q_f1']:.4f}"
    )
    print(f"[eval][step {global_step}] per-i accuracy: {per_i_acc}")
    print(f"[eval][step {global_step}] per-i q-confidence: {per_i_q}")
    print(f"[eval][step {global_step}] per-i q-F1(@{cfg.q_stop_threshold}): {per_i_qf1}")


def _run_lm_eval_at_checkpoint(
    combined_model: LDSForCausalLM,
    accelerator: Accelerator,
    cfg: TrainConfig,
    global_step: int,
    epoch: int,
) -> None:
    """Run lm-evaluation-harness at checkpoint save time and log results."""
    combined_model.eval()

    orig_reasoning = combined_model.reasoning
    orig_q_head = combined_model.q_head
    combined_model.reasoning = _unwrap_ddp(orig_reasoning)
    combined_model.q_head = _unwrap_ddp(orig_q_head)

    lm_eval_tasks = None
    if cfg.lm_eval_tasks:
        lm_eval_tasks = [task.strip() for task in cfg.lm_eval_tasks.split(",") if task.strip()]

    limit = cfg.lm_eval_limit if cfg.lm_eval_limit > 0 else None
    num_fewshot = cfg.lm_eval_num_fewshot if cfg.lm_eval_num_fewshot >= 0 else None

    if accelerator.is_main_process:
        task_names = lm_eval_tasks or DEFAULT_TASKS
        print(f"[lm-eval][step {global_step}] Running evaluation on: {task_names}")

    results = run_lm_eval_for_training(
        combined_model=combined_model,
        tasks=lm_eval_tasks,
        batch_size=cfg.lm_eval_batch_size,
        max_length=cfg.lm_eval_max_length,
        limit=limit,
        num_fewshot=num_fewshot,
        device=accelerator.device,
    )

    if accelerator.is_main_process:
        print(format_results_log(results, prefix=f"[lm-eval][step {global_step}] "))

        if cfg.eval_jsonl_path:
            append_jsonl(
                cfg.eval_jsonl_path,
                {
                    "type": "lm_eval",
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    **results,
                },
            )

        if cfg.eval_responses_dir:
            os.makedirs(cfg.eval_responses_dir, exist_ok=True)
            outpath = os.path.join(cfg.eval_responses_dir, f"lm_eval_step_{global_step}.json")
            with open(outpath, "w", encoding="utf-8") as handle:
                json.dump({"global_step": global_step, **results}, handle, indent=2, ensure_ascii=False)

        if cfg.wandb_enabled:
            wandb_lm: dict[str, int | float] = {}
            for task_name, task_metrics in results.get("results", {}).items():
                for metric_name, metric_value in task_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        wandb_lm[f"lm_eval/{task_name}/{metric_name}"] = metric_value
            if wandb_lm:
                wandb.log(wandb_lm, step=global_step)

    combined_model.reasoning = orig_reasoning
    combined_model.q_head = orig_q_head
    combined_model.train()


@torch.no_grad()
def evaluate(
    combined_model: LDSForCausalLM,
    dataloader: DataLoader,
    accelerator: Accelerator,
    n_supervision: int = 8,
    max_responses: int = 0,
    q_threshold: float = 0.5,
) -> dict:
    """Evaluate the model and return aggregated and per-supervision-step metrics."""
    combined_model.eval()

    orig_reasoning = combined_model.reasoning
    orig_q_head = combined_model.q_head
    combined_model.reasoning = _unwrap_ddp(orig_reasoning)
    combined_model.q_head = _unwrap_ddp(orig_q_head)

    tokenizer = combined_model.tokenizer
    device = accelerator.device
    total_tracker = MetricsTracker()
    per_i = [MetricsTracker() for _ in range(n_supervision)]

    total_q_counts = BinaryClassificationCounts()
    per_i_q_counts = [BinaryClassificationCounts() for _ in range(n_supervision)]

    collect_responses = max_responses != 0
    responses: list[dict] = []
    response_count = 0

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)
        labels = labels.to(device)

        with accelerator.autocast():
            hidden_states, position_embeddings, position_ids, cache_position, _ = combined_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        kwargs = combined_model._reasoning_kwargs(
            position_embeddings,
            position_ids,
            cache_position,
            attention_mask,
            hidden_states=hidden_states,
        )

        for n_step in range(n_supervision):
            hidden_states, q_logit = combined_model.run_final_with_q(hidden_states, **kwargs)
            logits = combined_model.decoder(hidden_states=hidden_states, **kwargs)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            y_hat = shift_logits.argmax(dim=-1)
            target_q, token_acc = compute_q_target(y_hat=y_hat, y_true=shift_labels)
            target_q = target_q.to(dtype=q_logit.dtype)
            q_loss = F.binary_cross_entropy_with_logits(q_logit, target_q)
            q_hat_val = torch.sigmoid(q_logit).mean().item()
            loss = lm_loss + q_loss

            update_q_metric_counts(per_i_q_counts[n_step], q_logit, target_q, q_threshold)

            per_i[n_step].update(
                loss=loss.item(),
                lm_loss=lm_loss.item(),
                q_loss=q_loss.item(),
                token_acc=token_acc,
                q_hat=q_hat_val,
            )

            if n_step == n_supervision - 1:
                update_q_metric_counts(total_q_counts, q_logit, target_q, q_threshold)
                total_tracker.update(
                    loss=loss.item(),
                    lm_loss=lm_loss.item(),
                    q_loss=q_loss.item(),
                    token_acc=token_acc,
                    q_hat=q_hat_val,
                )

                if collect_responses and (max_responses < 0 or response_count < max_responses):
                    batch_size = input_ids.size(0)
                    for batch_index in range(batch_size):
                        if 0 < max_responses <= response_count:
                            break

                        responses.append(
                            build_eval_response(
                                tokenizer=tokenizer,
                                input_ids=input_ids[batch_index],
                                attention_mask=attention_mask[batch_index],
                                shift_labels=shift_labels[batch_index],
                                y_hat=y_hat[batch_index],
                                q_logit=q_logit[batch_index],
                                index=response_count,
                            )
                        )
                        response_count += 1

            hidden_states = hidden_states.detach()

    totals = total_tracker.summarize()

    per_i_metrics = []
    for index, tracker in enumerate(per_i):
        per_i_metrics.append(summarize_eval_step(tracker, per_i_q_counts[index], index))

    combined_model.reasoning = orig_reasoning
    combined_model.q_head = orig_q_head

    total_q_metrics = summarize_q_metric_counts(total_q_counts)

    result = {
        "loss": totals["loss"],
        "lm_loss": totals["lm_loss"],
        "q_loss": totals["q_loss"],
        "token_acc": totals["token_acc"],
        "q_hat_mean": totals["q_hat"],
        **total_q_metrics,
        "per_i": per_i_metrics,
    }
    if collect_responses:
        result["responses"] = responses
    return result


@torch.no_grad()
def _collect_reasoning_step_losses(
    combined_model: LDSForCausalLM,
    accelerator: Accelerator,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
    n_reasoning_steps: int,
) -> list[dict[str, float]]:
    """Collect diagnostic losses for every reasoning step on the current batch."""
    with accelerator.autocast():
        hidden_states, position_embeddings, position_ids, cache_position, _ = combined_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    plateau_kwargs = combined_model._reasoning_kwargs(
        position_embeddings,
        position_ids,
        cache_position,
        attention_mask,
        hidden_states=hidden_states,
    )

    step_metrics: list[dict[str, float]] = []
    for step_index in range(n_reasoning_steps):
        hidden_states_old = hidden_states.detach()
        hidden_states, q_logit = combined_model.run_final_with_q(hidden_states_old, **plateau_kwargs)

        logits_old = combined_model.decoder(hidden_states=hidden_states_old, **plateau_kwargs)
        logits = combined_model.decoder(hidden_states=hidden_states, **plateau_kwargs)
        shift_logits_old = logits_old[:, :-1, :].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        lm_loss_prev = F.cross_entropy(
            shift_logits_old.view(-1, shift_logits_old.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss_mono = torch.nn.SiLU()(lm_loss - lm_loss_prev)
        y_hat = shift_logits.argmax(dim=-1)
        target_q, token_acc = compute_q_target(y_hat=y_hat, y_true=shift_labels)
        target_q = target_q.to(dtype=q_logit.dtype, device=q_logit.device)
        q_loss = F.binary_cross_entropy_with_logits(q_logit, target_q)
        loss = lm_loss + cfg.beta * loss_mono + q_loss

        step_metrics.append(
            {
                "step": step_index,
                "loss": loss.item(),
                "lm_loss": lm_loss.item(),
                "q_loss": q_loss.item(),
                "mono_loss": loss_mono.item(),
                "token_acc": token_acc,
                "q_hat": torch.sigmoid(q_logit).mean().item(),
            }
        )
        hidden_states = hidden_states.detach()

    return step_metrics


def _build_train_log_record(
    running: MetricsTracker,
    per_i_running: list[MetricsTracker],
    scheduler: torch.optim.lr_scheduler.LambdaLR | ApproximateTokenCosineScheduler | None,
    cfg: TrainConfig,
    global_step: int,
    epoch: int,
    reasoning_step_losses: list[dict[str, float]] | None = None,
) -> dict[str, float | int]:
    """Build the structured training log record for JSONL and W&B."""
    train_summary = running.summarize()
    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else cfg.learning_rate
    record: dict[str, float | int] = {
        "global_step": global_step,
        "epoch": epoch + 1,
        "loss": train_summary["loss"],
        "lm_loss": train_summary["lm_loss"],
        "q_loss": train_summary["q_loss"],
        "token_acc": train_summary["token_acc"],
        "q_hat_mean": train_summary["q_hat"],
        "lr": current_lr,
    }
    for index, tracker in enumerate(per_i_running):
        record[f"lm_loss_sup{index}"] = tracker.summarize()["lm_loss"]
    if reasoning_step_losses:
        for step_metrics in reasoning_step_losses:
            step = int(step_metrics["step"])
            record[f"loss_step{step}"] = step_metrics["loss"]
            record[f"lm_loss_step{step}"] = step_metrics["lm_loss"]
            record[f"q_loss_step{step}"] = step_metrics["q_loss"]
            record[f"mono_loss_step{step}"] = step_metrics["mono_loss"]
            record[f"token_acc_step{step}"] = step_metrics["token_acc"]
            record[f"q_hat_step{step}"] = step_metrics["q_hat"]
    return record


def _format_reasoning_step_log(
    reasoning_step_losses: list[dict[str, float]],
    key: str,
    prefix: str,
) -> str:
    """Format one reasoning-step metric line for console logging."""
    return prefix + " ".join(
        f"step{int(step_metrics['step'])}={step_metrics[key]:.4f}"
        for step_metrics in reasoning_step_losses
    )


def _build_wandb_train_record(record: dict[str, float | int]) -> dict[str, float | int]:
    """Convert flat train metrics into a W&B-friendly namespaced payload."""
    wandb_record: dict[str, float | int] = {}
    for key, value in record.items():
        if key in {"global_step", "epoch"}:
            continue

        if key.startswith((
            "loss_step",
            "lm_loss_step",
            "q_loss_step",
            "mono_loss_step",
            "token_acc_step",
            "q_hat_step",
        )):
            metric_name, _, step_suffix = key.rpartition("_step")
            wandb_record[f"train/reasoning_steps/step_{step_suffix}/{metric_name}"] = value
            continue

        wandb_record[f"train/{key}"] = value

    return wandb_record


def _log_training_progress(
    accelerator: Accelerator,
    cfg: TrainConfig,
    combined_model: LDSForCausalLM,
    running: MetricsTracker,
    per_i_running: list[MetricsTracker],
    scheduler: torch.optim.lr_scheduler.LambdaLR | ApproximateTokenCosineScheduler | None,
    global_step: int,
    epoch: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Flush periodic train metrics to console and configured backends."""
    should_log = (
        accelerator.is_main_process
        and running.steps > 0
        and global_step % cfg.log_interval == 0
    )
    if not should_log:
        return

    reasoning_step_losses: list[dict[str, float]] | None = None
    if not _uses_sharded_parameters(accelerator):
        reasoning_step_losses = _collect_reasoning_step_losses(
            combined_model=combined_model,
            accelerator=accelerator,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            cfg=cfg,
            n_reasoning_steps=cfg.n_reasoning_steps,
        )

    record = _build_train_log_record(
        running=running,
        per_i_running=per_i_running,
        scheduler=scheduler,
        cfg=cfg,
        global_step=global_step,
        epoch=epoch,
        reasoning_step_losses=reasoning_step_losses,
    )
    per_i_str = " ".join(
        f"lm_sup{index}={tracker.summarize()['lm_loss']:.4f}"
        for index, tracker in enumerate(per_i_running)
    )
    header = running.format_log(prefix=f"[train][epoch {epoch + 1}]", step=global_step)
    print(header + f" | {per_i_str}")
    if reasoning_step_losses:
        print(_format_reasoning_step_log(reasoning_step_losses, "loss", f"[train][step {global_step}] reasoning loss: "))
        print(_format_reasoning_step_log(reasoning_step_losses, "lm_loss", f"[train][step {global_step}] reasoning lm: "))
        print(_format_reasoning_step_log(reasoning_step_losses, "q_loss", f"[train][step {global_step}] reasoning q: "))
        print(_format_reasoning_step_log(reasoning_step_losses, "mono_loss", f"[train][step {global_step}] reasoning mono: "))

    if cfg.train_jsonl_path:
        append_jsonl(cfg.train_jsonl_path, record)

    if cfg.wandb_enabled:
        wandb_record = _build_wandb_train_record(record)
        wandb.log(wandb_record, step=global_step)

    running.reset()
    for tracker in per_i_running:
        tracker.reset()


def _maybe_save_checkpoint(
    combined_model: LDSForCausalLM,
    accelerator: Accelerator,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
) -> None:
    """Save a checkpoint and optionally run lm-eval when the interval matches."""
    should_save = cfg.save_interval > 0 and global_step % cfg.save_interval == 0
    if not should_save:
        return

    # All ranks must participate in state_dict() for FSDP/DeepSpeed sharding,
    # but only rank 0 actually writes files to disk.
    state_dict = accelerator.get_state_dict(combined_model)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(combined_model)
        unwrapped.save_checkpoint(
            checkpoint_dir=cfg.checkpoint_dir,
            global_step=global_step,
            epoch=epoch,
            optimizer=optimizer,
            max_checkpoints=cfg.max_checkpoints,
            state_dict_override=state_dict,
        )
    accelerator.wait_for_everyone()

    if cfg.lm_eval_at_save:
        if accelerator.is_main_process:
            _run_lm_eval_at_checkpoint(combined_model, accelerator, cfg, global_step, epoch)
        accelerator.wait_for_everyone()


def _maybe_push_to_hub(
    combined_model: LDSForCausalLM,
    accelerator: Accelerator,
    cfg: TrainConfig,
    global_step: int,
) -> None:
    """Push the model to HuggingFace Hub when configured."""
    should_push = (
        cfg.hub_push_interval > 0
        and cfg.hub_repo_id
        and global_step % cfg.hub_push_interval == 0
    )
    if not should_push:
        return

    if accelerator.is_main_process:
        print(f"[Step {global_step}] Pushing to HuggingFace Hub: {cfg.hub_repo_id}")
        combined_model.push_to_hub(
            cfg.hub_repo_id,
            commit_message=f"step {global_step}",
            private=cfg.hub_private,
        )
        print(f"[Step {global_step}] Push complete.")
    accelerator.wait_for_everyone()


def _maybe_run_validation(
    combined_model: LDSForCausalLM,
    eval_dataloader: DataLoader | None,
    accelerator: Accelerator,
    cfg: TrainConfig,
    global_step: int,
    epoch: int,
) -> None:
    """Run the validation pass when the configured interval matches."""
    should_evaluate = (
        eval_dataloader is not None
        and cfg.eval_interval > 0
        and global_step % cfg.eval_interval == 0
    )
    if not should_evaluate:
        return

    assert eval_dataloader is not None
    accelerator.wait_for_everyone()
    combined_model.eval()
    _run_and_log_evaluation(combined_model, eval_dataloader, accelerator, cfg, global_step, epoch)
    combined_model.train()
    accelerator.wait_for_everyone()


def _run_periodic_tasks(
    combined_model: LDSForCausalLM,
    accelerator: Accelerator,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | ApproximateTokenCosineScheduler | None,
    running: MetricsTracker,
    per_i_running: list[MetricsTracker],
    eval_dataloader: DataLoader | None,
    global_step: int,
    epoch: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Handle non-loss side effects that are scheduled on training steps."""
    _log_training_progress(
        accelerator=accelerator,
        cfg=cfg,
        combined_model=combined_model,
        running=running,
        per_i_running=per_i_running,
        scheduler=scheduler,
        global_step=global_step,
        epoch=epoch,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    _maybe_save_checkpoint(
        combined_model=combined_model,
        accelerator=accelerator,
        cfg=cfg,
        optimizer=optimizer,
        global_step=global_step,
        epoch=epoch,
    )
    _maybe_push_to_hub(
        combined_model=combined_model,
        accelerator=accelerator,
        cfg=cfg,
        global_step=global_step,
    )
    _maybe_run_validation(
        combined_model=combined_model,
        eval_dataloader=eval_dataloader,
        accelerator=accelerator,
        cfg=cfg,
        global_step=global_step,
        epoch=epoch,
    )


def train_with_deep_supervision(
    combined_model: LDSForCausalLM,
    accelerator: Accelerator,
    dataloader: DataLoader,
    cfg: TrainConfig,
    eval_dataloader: DataLoader | None = None,
    estimated_steps_per_epoch: int | None = None,
    estimated_train_samples: int | None = None,
) -> None:
    """Deep-supervision training for ReasoningBlock with Q-head."""
    setup = _setup_training(
        combined_model=combined_model,
        accelerator=accelerator,
        dataloader=dataloader,
        cfg=cfg,
        eval_dataloader=eval_dataloader,
        estimated_steps_per_epoch=estimated_steps_per_epoch,
        estimated_train_samples=estimated_train_samples,
    )

    running = MetricsTracker()
    epoch_tracker = MetricsTracker()
    per_i_running = [MetricsTracker() for _ in range(cfg.n_supervision)]
    pending_token_scheduler_tokens = 0

    global_step = setup.global_step
    for epoch in range(setup.start_epoch, cfg.epochs):
        running.reset()
        epoch_tracker.reset()
        for tracker in per_i_running:
            tracker.reset()

        progress_total = (
            _estimate_optimizer_updates(setup.steps_per_epoch, cfg)
            if setup.steps_per_epoch > 0 else None
        )
        progress_initial = (
            _estimate_optimizer_updates(setup.start_step, cfg)
            if epoch == setup.start_epoch else 0
        )
        progress_bar = None
        if accelerator.is_main_process:
            progress_bar = tqdm(
                total=progress_total,
                initial=progress_initial,
                desc=f"Epoch {epoch + 1}/{cfg.epochs}",
                leave=False,
                dynamic_ncols=True,
            )

        step_in_epoch = 0
        try:
            for input_ids, attention_mask, labels in setup.dataloader:
                if epoch == setup.start_epoch and step_in_epoch < setup.start_step:
                    step_in_epoch += 1
                    continue

                input_ids = input_ids.to(accelerator.device)
                attention_mask = attention_mask.to(accelerator.device)
                labels = labels.to(accelerator.device)

                with torch.no_grad():
                    hidden_states, position_embeddings, position_ids, cache_position, _ = combined_model.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                plateau_kwargs = combined_model._reasoning_kwargs(
                    position_embeddings,
                    position_ids,
                    cache_position,
                    attention_mask,
                    hidden_states=hidden_states,
                )

                supervised_indices = sorted(
                    random.sample(range(cfg.n_reasoning_steps), cfg.n_supervision)
                )
                supervised_set = set(supervised_indices)
                
                
                sup_idx = 0

                for n_step in range(cfg.n_reasoning_steps):
                    if n_step in supervised_set:
                        with accelerator.accumulate(combined_model.reasoning, combined_model.q_head):
                            loss, token_acc, q_hat, hidden_states, lm_loss, q_loss = _compute_supervision_loss(
                                combined_model=combined_model,
                                hidden_states=hidden_states,
                                labels=labels,
                                cfg=cfg,
                                plateau_kwargs=plateau_kwargs,
                            )

                            accelerator.backward(loss)
                            setup.optimizer.step()
                            if setup.scheduler is not None:
                                if _is_token_scheduler(setup.scheduler):
                                    pending_token_scheduler_tokens += _approx_tokens_per_batch(
                                        input_ids,
                                        cfg,
                                        accelerator,
                                    )
                                    if accelerator.sync_gradients:
                                        setup.scheduler.step(pending_token_scheduler_tokens)
                                        pending_token_scheduler_tokens = 0
                                else:
                                    cast(torch.optim.lr_scheduler.LambdaLR, setup.scheduler).step()
                            setup.optimizer.zero_grad()

                            if progress_bar is not None and accelerator.sync_gradients:
                                progress_bar.update(1)
                                progress_bar.set_postfix(
                                    loss=f"{loss.item():.4f}",
                                    lm=f"{lm_loss.item():.4f}",
                                    q=f"{q_loss.item():.4f}",
                                    refresh=False,
                                )

                            hidden_states = hidden_states.detach()
                            per_i_running[sup_idx].update(
                                loss=loss.item(),
                                lm_loss=lm_loss.item(),
                                q_loss=q_loss.item(),
                                token_acc=token_acc,
                                q_hat=q_hat.mean().item(),
                            )
                        sup_idx += 1
                    else:
                        with torch.no_grad():
                            hidden_states = combined_model.reasoning(
                                hidden_states=hidden_states,
                                **plateau_kwargs,
                            )
                            hidden_states = hidden_states.detach()

                global_step += 1

                epoch_tracker.update(
                    loss=loss.item(),
                    lm_loss=lm_loss.item(),
                    q_loss=q_loss.item(),
                    token_acc=token_acc,
                    q_hat=q_hat.mean().item(),
                )
                running.update(
                    loss=loss.item(),
                    lm_loss=lm_loss.item(),
                    q_loss=q_loss.item(),
                    token_acc=token_acc,
                    q_hat=q_hat.mean().item(),
                )

                _run_periodic_tasks(
                    combined_model=combined_model,
                    accelerator=accelerator,
                    cfg=cfg,
                    optimizer=setup.optimizer,
                    scheduler=setup.scheduler,
                    running=running,
                    per_i_running=per_i_running,
                    eval_dataloader=setup.eval_dataloader,
                    global_step=global_step,
                    epoch=epoch,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                step_in_epoch += 1
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if accelerator.is_main_process:
            summary = epoch_tracker.summarize()
            print(
                f"[Epoch {epoch + 1}/{cfg.epochs}] "
                f"loss={summary['loss']:.4f} "
                f"lm={summary['lm_loss']:.4f} "
                f"q={summary['q_loss']:.4f}"
            )


def run_training_entrypoint(
    cfg: TrainConfig,
    dataloader_factory: DataLoaderFactory,
    dataset_log_prefix: str,
) -> None:
    """Run the shared bootstrap flow for a training entrypoint."""
    set_seed(cfg.seed)
    configure_external_services()

    accelerator = create_accelerator(cfg)
    initialize_wandb(cfg, accelerator)

    combined_model = prepare_model(
        cfg.model_name,
        encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers,
        n_reasoning_steps=cfg.n_reasoning_steps,
        from_hub=cfg.from_hub,
    )
    combined_model = combined_model.to(accelerator.device)

    train_dataloader, eval_dataloader, n_train, n_eval = dataloader_factory(
        cfg,
        combined_model.tokenizer,
        accelerator,
    )
    log_dataset_overview(cfg, accelerator, n_train, n_eval, dataset_log_prefix)
    log_lm_eval_plan(cfg, accelerator)

    estimated_steps = estimate_steps_per_epoch(
        n_train=n_train,
        batch_size=cfg.batch_size,
        num_processes=accelerator.num_processes,
    )
    train_with_deep_supervision(
        combined_model=combined_model,
        accelerator=accelerator,
        dataloader=train_dataloader,
        cfg=cfg,
        eval_dataloader=eval_dataloader,
        estimated_steps_per_epoch=estimated_steps,
        estimated_train_samples=n_train if n_train > 0 else None,
    )

    save_final_model(combined_model, cfg, accelerator)
    push_final_model_to_hub(combined_model, cfg, accelerator)
    finalize_wandb(cfg, accelerator)
    accelerator.end_training()