"""Metric tracking and logging utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Scalar metric helpers
# ---------------------------------------------------------------------------

def compute_q_target(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Compute per-sample Q target and scalar token accuracy.

    Args:
        y_hat: Predicted token ids ``(B, T)``.
        y_true: Ground-truth token ids ``(B, T)`` with ``-100`` for padding.

    Returns:
        target_q: Per-sample accuracy ``(B,)``.
        token_acc: Scalar micro-averaged token accuracy.
    """
    valid_mask = y_true.ne(-100)
    token_correct = (y_hat.eq(y_true) & valid_mask).float()
    valid_count = valid_mask.float().sum(dim=1).clamp(min=1.0)
    target_q = token_correct.sum(dim=1) / valid_count
    token_acc = (token_correct.sum() / valid_mask.float().sum().clamp(min=1.0)).item()
    return target_q, token_acc


def append_jsonl(file_path: str, record: dict) -> None:
    """Append a single JSON object as a line to *file_path*."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class BinaryClassificationCounts:
    """Accumulate binary classification counts for precision/recall/F1."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update counts from boolean prediction and target tensors."""
        self.tp += (predicted & target).sum().item()
        self.fp += (predicted & ~target).sum().item()
        self.fn += (~predicted & target).sum().item()

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / max(self.precision + self.recall, 1e-8)


def update_q_metric_counts(
    counts: BinaryClassificationCounts,
    q_logits: torch.Tensor,
    target_q: torch.Tensor,
    threshold: float,
) -> None:
    """Update classification counts for q-score predictions at the given threshold."""
    predicted = torch.sigmoid(q_logits) >= threshold
    target = target_q >= threshold
    counts.update(predicted=predicted, target=target)


def summarize_q_metric_counts(counts: BinaryClassificationCounts) -> dict[str, float]:
    """Convert binary classification counts into scalar metrics."""
    return {
        "q_f1": counts.f1,
        "q_precision": counts.precision,
        "q_recall": counts.recall,
    }


def summarize_eval_step(
    tracker: "MetricsTracker",
    counts: BinaryClassificationCounts,
    index: int,
) -> dict[str, float | int]:
    """Build the per-supervision-step evaluation summary."""
    summary: dict[str, float | int] = tracker.summarize()
    summary["i"] = index
    summary["count"] = tracker.steps
    summary.update(summarize_q_metric_counts(counts))
    return summary


def build_eval_response(
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    shift_labels: torch.Tensor,
    y_hat: torch.Tensor,
    q_logit: torch.Tensor,
    index: int,
    max_input_chars: int = 500,
) -> dict[str, str | float | int]:
    """Build one decoded evaluation sample for JSONL logging."""
    valid_mask = shift_labels.ne(-100)
    ground_truth = tokenizer.decode(shift_labels[valid_mask], skip_special_tokens=True)
    prediction = tokenizer.decode(y_hat[valid_mask], skip_special_tokens=True)
    input_text = tokenizer.decode(input_ids[attention_mask], skip_special_tokens=True)

    sample_correct = (y_hat.eq(shift_labels) & valid_mask).sum().item()
    sample_total = valid_mask.sum().item()
    sample_acc = sample_correct / max(sample_total, 1)

    return {
        "index": index,
        "input": input_text[:max_input_chars],
        "ground_truth": ground_truth,
        "prediction": prediction,
        "token_acc": round(sample_acc, 4),
        "q_hat": round(torch.sigmoid(q_logit).item(), 4),
    }


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Accumulates training metrics and provides averaged summaries.

    Usage::

        tracker = MetricsTracker()
        for batch in dataloader:
            ...
            tracker.update(loss=..., lm_loss=..., q_loss=...,
                           token_acc=..., q_hat=...)
        summary = tracker.summarize_and_reset()
    """

    __slots__ = ("_loss", "_lm_loss", "_q_loss", "_token_acc", "_q_hat", "_steps")

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._loss = 0.0
        self._lm_loss = 0.0
        self._q_loss = 0.0
        self._token_acc = 0.0
        self._q_hat = 0.0
        self._steps = 0

    def update(
        self,
        loss: float,
        lm_loss: float,
        q_loss: float,
        token_acc: float,
        q_hat: float,
    ) -> None:
        self._loss += loss
        self._lm_loss += lm_loss
        self._q_loss += q_loss
        self._token_acc += token_acc
        self._q_hat += q_hat
        self._steps += 1

    # ------------------------------------------------------------------

    @property
    def steps(self) -> int:
        return self._steps

    def summarize(self) -> dict[str, float]:
        """Return averaged metrics *without* resetting."""
        denom = max(1, self._steps)
        return {
            "loss": self._loss / denom,
            "lm_loss": self._lm_loss / denom,
            "q_loss": self._q_loss / denom,
            "token_acc": self._token_acc / denom,
            "q_hat": self._q_hat / denom,
        }

    def summarize_and_reset(self) -> dict[str, float]:
        """Return averaged metrics and reset all accumulators."""
        result = self.summarize()
        self.reset()
        return result

    def format_log(self, prefix: str = "", step: int | None = None) -> str:
        """Return a formatted single-line log string."""
        s = self.summarize()
        parts: list[str] = []
        if prefix:
            parts.append(prefix)
        if step is not None:
            parts.append(f"step {step}")
        header = "[" + " ".join(parts) + "] " if parts else ""
        return (
            f"{header}"
            f"loss={s['loss']:.4f} "
            f"lm={s['lm_loss']:.4f} "
            f"q={s['q_loss']:.4f} "
            f"token_acc={s['token_acc']:.4f} "
            f"q_hat={s['q_hat']:.4f}"
        )
