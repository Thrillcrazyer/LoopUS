"""Shared CLI helpers for DDD training entrypoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainCliDefaults:
    """Default values that vary between training entrypoints."""

    description: str
    train_split: str
    eval_jsonl_path: str
    train_jsonl_path: str
    checkpoint_dir: str
    eval_responses_dir: str
    wandb_project: str


def build_train_arg_parser(defaults: TrainCliDefaults) -> argparse.ArgumentParser:
    """Build the shared training parser with entrypoint-specific defaults."""
    parser = argparse.ArgumentParser(description=defaults.description)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-dataset", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument(
        "--train-config",
        type=str,
        default="",
        help="Dataset config/subset name (e.g. 'sample-10BT' for fineweb-edu)",
    )
    parser.add_argument("--train-split", type=str, default=defaults.train_split)
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=200_000,
        help="Target sample budget (-1 or 0 keeps streaming dataset unbounded)",
    )
    parser.add_argument(
        "--train-max-tokens",
        type=int,
        default=-1,
        help="Target token budget; overrides --train-max-samples when > 0 "
        "(computed as train_max_tokens // max_length)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--n-supervision", type=int, default=2)
    parser.add_argument(
        "--n-reasoning-steps",
        type=int,
        default=100,
        help="Total reasoning iterations per batch; n-supervision of these get loss",
    )
    parser.add_argument("--t-recursion", type=int, default=5)
    parser.add_argument(
        "--n-latent",
        type=int,
        default=6,
        help="Number of inner latent recursion iterations per deep recursion step",
    )
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Run validation every N steps (-1 or 0 to disable)",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-jsonl-path", type=str, default=defaults.eval_jsonl_path)
    parser.add_argument(
        "--train-jsonl-path",
        type=str,
        default=defaults.train_jsonl_path,
        help="Path to JSONL file for periodic training metrics (empty to disable)",
    )
    parser.add_argument("--q-stop-threshold", type=float, default=0.55)
    parser.add_argument("--q-stop-mode", type=str, default="all", choices=["all", "any"])
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight for the monotonic-improvement penalty",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Weight for the hidden-state/gradient alignment penalty",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep (0 for unlimited)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=defaults.checkpoint_dir,
        help="Directory to save/load checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint directory to resume from (e.g. checkpoints/step_1000)",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=-1,
        help="Override the batch skip count on resume (-1 = auto from checkpoint, 0 = no skip)",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--from-hub",
        type=str,
        default="",
        help="Load a pre-trained LDS model from HF Hub repo id or local path "
        "(e.g. 'Thrillcrazyer/LDS_1.8B') instead of building from base model",
    )
    parser.add_argument(
        "--encoder-layers",
        type=str,
        default=None,
        help="Encoder layers: '2' = 0..2, '0,1,2' = explicit, '0-2' or '0..2' = range, 'none' = no encoder blocks",
    )
    parser.add_argument(
        "--decoder-layers",
        type=str,
        default=None,
        help="Decoder layers: '13-' or '13..' = 13..last, '13-27' or '13..27' = range, '25,26,27' = explicit, 'none' = no decoder blocks",
    )
    parser.add_argument(
        "--lm-eval-at-save",
        action="store_true",
        help="Run lm-evaluation-harness benchmarks every time a checkpoint is saved",
    )
    parser.add_argument(
        "--lm-eval-tasks",
        type=str,
        default="",
        help="Comma-separated lm-eval task names "
        "(default: mmlu,hellaswag,arc_easy,arc_challenge,piqa,"
        "winogrande,lambada_openai,wikitext,boolq,openbookqa)",
    )
    parser.add_argument(
        "--lm-eval-limit",
        type=int,
        default=200,
        help="Limit samples per task during training eval (0 = all)",
    )
    parser.add_argument(
        "--lm-eval-batch-size",
        type=int,
        default=4,
        help="Batch size for lm-eval evaluation",
    )
    parser.add_argument(
        "--lm-eval-max-length",
        type=int,
        default=2048,
        help="Max sequence length for lm-eval",
    )
    parser.add_argument(
        "--lm-eval-num-fewshot",
        type=int,
        default=-1,
        help="Number of few-shot examples for lm-eval (-1 = task default)",
    )
    parser.add_argument(
        "--eval-responses-dir",
        type=str,
        default=defaults.eval_responses_dir,
        help="Directory to save decoded model responses during evaluation",
    )
    parser.add_argument(
        "--eval-max-responses",
        type=int,
        default=50,
        help="Max number of response samples to save per eval step (0 = all)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Number of warmup steps for LR scheduler (0 to disable scheduler)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "cosine_tokens", "none"],
        help="LR scheduler type",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Use pinned memory for DataLoader",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default="",
        help="HuggingFace Hub repo id to push checkpoints to (e.g. 'user/model')",
    )
    parser.add_argument(
        "--hub-push-interval",
        type=int,
        default=0,
        help="Push model to HuggingFace Hub every N steps (0 to disable)",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        default=False,
        help="Create the Hub repo as private",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=defaults.wandb_project,
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
        help="W&B run name (empty = auto-generated)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="W&B entity (team or username)",
    )
    return parser