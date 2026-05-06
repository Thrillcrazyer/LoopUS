"""SFT training entrypoint for the decomposed DDD model."""

from __future__ import annotations

from accelerate import Accelerator

from training_cli import TrainCliDefaults, build_train_arg_parser
from training_runtime import resolve_sample_budget, run_training_entrypoint
from utils.common import TrainConfig
from utils.data import create_sft_streaming_dataloaders


CLI_DEFAULTS = TrainCliDefaults(
    description="SFT training for decomposed model with deep supervision",
    train_split="train_sft",
    eval_jsonl_path="logs/sft_eval_metrics.jsonl",
    train_jsonl_path="logs/sft_train_metrics.jsonl",
    checkpoint_dir="checkpoints_sft",
    eval_responses_dir="logs/sft_eval_responses",
    wandb_project="DDD-SFT",
)


def parse_args():
    return build_train_arg_parser(CLI_DEFAULTS).parse_args()


def _build_dataloaders(cfg: TrainConfig, tokenizer, accelerator: Accelerator):
    max_samples = resolve_sample_budget(cfg, accelerator)
    train_alignment_samples = (
        cfg.batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes
    )
    return create_sft_streaming_dataloaders(
        tokenizer=tokenizer,
        dataset_name=cfg.train_dataset,
        split=cfg.train_split,
        config=cfg.train_config or None,
        max_samples=max_samples,
        val_ratio=cfg.val_ratio,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        seed=cfg.seed,
        train_alignment_samples=train_alignment_samples,
    )


def main() -> None:
    cfg = TrainConfig.from_args(parse_args())
    run_training_entrypoint(
        cfg=cfg,
        dataloader_factory=_build_dataloaders,
        dataset_log_prefix="SFT streaming",
    )


if __name__ == "__main__":
    main()