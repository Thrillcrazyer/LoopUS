"""Common utilities shared across training and evaluation."""

import random
from dataclasses import dataclass, field

import numpy as np
import torch


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_layer_indices(raw: str | None) -> list[int] | None:
    """Parse a layer-index specification string.

    Supported formats:
    - ``"0,1,2"`` → ``[0, 1, 2]`` (explicit list)
    - ``"2"`` → ``[0, 1, 2]`` (single int → 0 .. N inclusive)
    - ``"0..2"`` → ``[0, 1, 2]`` (closed range; alias for ``"0-2"``)
    - ``"13-"`` → ``[-13]`` (open-ended; expanded later with num_layers)
    - ``"13.."`` → ``[-13]`` (open-ended; alias for ``"13-"``)
    - ``"3-7"`` → ``[3, 4, 5, 6, 7]`` (closed range)
    - ``"none"`` / ``"empty"`` / ``"[]"`` → ``[]`` (explicitly no layers)

    A single trailing ``-`` (e.g. ``"13-"``) produces a *negative sentinel*
    ``[-start]`` which :class:`LDSConfig` expands to ``[start .. num_layers-1]``
    once the total number of layers is known.
    """
    if not raw:
        return None
    raw = raw.strip()

    if raw.lower() in {"none", "empty", "[]"}:
        return []

    raw = raw.replace("..", "-")

    # Open-ended range: "13-" means layer 13 to the last layer
    if raw.endswith("-") and "," not in raw:
        start = int(raw[:-1])
        return [-start]  # sentinel; expanded in LDSConfig

    # Closed range: "3-7"
    if "-" in raw and "," not in raw:
        parts = raw.split("-", 1)
        a, b = int(parts[0]), int(parts[1])
        return list(range(a, b + 1))

    # Comma-separated list or single integer
    indices = [int(x.strip()) for x in raw.split(",")]
    if len(indices) == 1:
        # Single int N → [0 .. N]
        return list(range(indices[0] + 1))
    return indices


@dataclass
class TrainConfig:
    """Consolidated training configuration."""

    seed: int = 42
    model_name: str = "Qwen/Qwen3-1.7B"
    train_dataset: str = "HuggingFaceH4/ultrachat_200k"
    train_config: str = ""
    train_split: str = "train_sft"
    train_max_samples: int = 200_000
    train_max_tokens: int = -1
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 5e-5
    max_length: int = 1024
    n_supervision: int = 2
    n_reasoning_steps: int = 100
    t_recursion: int = 5
    n_latent: int = 6
    val_ratio: float = 0.01
    eval_interval: int = 500
    log_interval: int = 100
    eval_jsonl_path: str = "logs/eval_metrics.jsonl"
    train_jsonl_path: str = "logs/train_metrics.jsonl"
    q_stop_threshold: float = 0.9
    q_stop_mode: str = "all"
    beta: float = 1.0
    gamma: float = 1.0
    save_interval: int = 1000
    max_checkpoints: int = 3
    checkpoint_dir: str = "checkpoints"
    resume: str = ""
    start_step: int = -1
    from_hub: str = ""
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    encoder_layers: list[int] | None = field(default=None)
    decoder_layers: list[int] | None = field(default=None)

    # MMLU evaluation at checkpoint save time (legacy — replaced by lm-eval)
    mmlu_eval_at_save: bool = False
    mmlu_dataset_name: str = "cais/mmlu"
    mmlu_subjects: str = ""
    mmlu_max_samples: int = 500
    mmlu_num_few_shot: int = 5
    mmlu_batch_size: int = 8

    # lm-evaluation-harness at checkpoint save time
    lm_eval_at_save: bool = False
    lm_eval_tasks: str = ""
    lm_eval_limit: int = 200
    lm_eval_batch_size: int = 4
    lm_eval_max_length: int = 2048
    lm_eval_num_fewshot: int = -1

    # Eval response saving
    eval_responses_dir: str = "logs/eval_responses"
    eval_max_responses: int = 50

    # Wiki-PPL evaluation at checkpoint save time (legacy — replaced by lm-eval)
    wiki_ppl_eval_at_save: bool = False
    wiki_ppl_dataset: str = "wikitext"
    wiki_ppl_config: str = "wikitext-2-raw-v1"
    wiki_ppl_split: str = "test"
    wiki_ppl_max_length: int = 1024
    wiki_ppl_stride: int = 512

    # LR Scheduler
    warmup_steps: int = 200
    lr_scheduler: str = "cosine"

    # DataLoader optimization
    num_workers: int = 4
    pin_memory: bool = True

    # HuggingFace Hub
    hub_repo_id: str = ""
    hub_push_interval: int = 0
    hub_private: bool = False

    # Weights & Biases
    wandb_enabled: bool = False
    wandb_project: str = "DDD"
    wandb_run_name: str = ""
    wandb_entity: str = ""

    def validate(self) -> None:
        """Raise ``ValueError`` on invalid settings."""
        if self.q_stop_mode not in {"all", "any"}:
            raise ValueError("q_stop_mode must be 'all' or 'any'")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        if self.t_recursion < 1:
            raise ValueError("t_recursion must be >= 1")
        if self.n_supervision > self.n_reasoning_steps:
            raise ValueError(
                f"n_supervision ({self.n_supervision}) must be <= "
                f"n_reasoning_steps ({self.n_reasoning_steps})"
            )

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        """Build a ``TrainConfig`` from an ``argparse.Namespace``."""
        cfg = cls(
            seed=args.seed,
            model_name=args.model_name,
            train_dataset=args.train_dataset,
            train_config=getattr(args, "train_config", ""),
            train_split=args.train_split,
            train_max_samples=args.train_max_samples,
            train_max_tokens=getattr(args, "train_max_tokens", -1),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            n_supervision=args.n_supervision,
            n_reasoning_steps=getattr(args, "n_reasoning_steps", 100),
            t_recursion=args.t_recursion,
            n_latent=args.n_latent,
            val_ratio=args.val_ratio,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            eval_jsonl_path=args.eval_jsonl_path,
            train_jsonl_path=getattr(args, "train_jsonl_path", "logs/train_metrics.jsonl"),
            q_stop_threshold=args.q_stop_threshold,
            q_stop_mode=args.q_stop_mode,
            beta=getattr(args, "beta", 1.0),
            gamma=getattr(args, "gamma", 1.0),
            save_interval=args.save_interval,
            max_checkpoints=args.max_checkpoints,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            start_step=getattr(args, "start_step", -1),
            from_hub=getattr(args, "from_hub", ""),
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            encoder_layers=parse_layer_indices(args.encoder_layers),
            decoder_layers=parse_layer_indices(args.decoder_layers),
            mmlu_eval_at_save=getattr(args, "mmlu_eval_at_save", False),
            mmlu_dataset_name=getattr(args, "mmlu_dataset_name", "cais/mmlu"),
            mmlu_subjects=getattr(args, "mmlu_subjects", ""),
            mmlu_max_samples=getattr(args, "mmlu_max_samples", 500),
            mmlu_num_few_shot=getattr(args, "mmlu_num_few_shot", 5),
            mmlu_batch_size=getattr(args, "mmlu_batch_size", 8),
            lm_eval_at_save=getattr(args, "lm_eval_at_save", False),
            lm_eval_tasks=getattr(args, "lm_eval_tasks", ""),
            lm_eval_limit=getattr(args, "lm_eval_limit", 200),
            lm_eval_batch_size=getattr(args, "lm_eval_batch_size", 4),
            lm_eval_max_length=getattr(args, "lm_eval_max_length", 2048),
            lm_eval_num_fewshot=getattr(args, "lm_eval_num_fewshot", -1),
            eval_responses_dir=getattr(args, "eval_responses_dir", "logs/eval_responses"),
            eval_max_responses=getattr(args, "eval_max_responses", 50),
            wiki_ppl_eval_at_save=getattr(args, "wiki_ppl_eval_at_save", False),
            wiki_ppl_dataset=getattr(args, "wiki_ppl_dataset", "wikitext"),
            wiki_ppl_config=getattr(args, "wiki_ppl_config", "wikitext-2-raw-v1"),
            wiki_ppl_split=getattr(args, "wiki_ppl_split", "test"),
            wiki_ppl_max_length=getattr(args, "wiki_ppl_max_length", 1024),
            wiki_ppl_stride=getattr(args, "wiki_ppl_stride", 512),
            warmup_steps=getattr(args, "warmup_steps", 200),
            lr_scheduler=getattr(args, "lr_scheduler", "cosine"),
            num_workers=getattr(args, "num_workers", 4),
            pin_memory=getattr(args, "pin_memory", True),
            hub_repo_id=getattr(args, "hub_repo_id", ""),
            hub_push_interval=getattr(args, "hub_push_interval", 0),
            hub_private=getattr(args, "hub_private", False),
            wandb_enabled=getattr(args, "wandb", False),
            wandb_project=getattr(args, "wandb_project", "DDD"),
            wandb_run_name=getattr(args, "wandb_run_name", ""),
            wandb_entity=getattr(args, "wandb_entity", ""),
        )
        cfg.validate()
        return cfg
