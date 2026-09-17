#!/usr/bin/env python3
"""Compare eval-mode reasoning-residual curves across LoopUS checkpoints.

Runs the same fixed batch through each checkpoint's reasoning recursion and
reports the per-step fixed-point residual. The diagnostic forces eval mode, so
it measures the attractor landscape training *shaped* rather than any noise
being injected — which is what distinguishes "NI changed the dynamics" from
"NI did nothing".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training_runtime import _collect_reasoning_step_losses
from utils.common import TrainConfig, set_seed
from utils.data import StreamingTokenDataset
from utils.inference import load_lds_model, resolve_device_and_dtype

METRICS = ("residual_rel", "residual", "state_norm", "lm_loss", "token_acc")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+",
                   help="LABEL=PATH entries, e.g. ctrl=checkpoints/LoopUS_NI0.0_RI0.0/final")
    p.add_argument("--n-reasoning-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--train-dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--train-config", type=str, default="CC-MAIN-2025-26")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--metric", type=str, default="residual_rel", choices=METRICS)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", type=str, default=None)
    return p.parse_args()


def default_label(path: str) -> str:
    parts = Path(path.rstrip("/")).parts
    name = parts[-1]
    # checkpoints/<run>/final -> "<run>"; hub repo "org/name" -> "name"
    if len(parts) > 1 and (name in {"final", "latest"} or name.startswith("step_")):
        return parts[-2]
    return name


def parse_checkpoints(raw: list[str]) -> list[tuple[str, str]]:
    entries = []
    for item in raw:
        label, sep, path = item.partition("=")
        if not sep:
            label, path = default_label(item), item

        local = Path(path)
        if local.is_dir():
            if not (local / "config.json").is_file():
                raise SystemExit(
                    f"{path} has no config.json — pass a save_pretrained-style dir "
                    "(a final/ dir needs config.json + tokenizer files copied in)."
                )
        elif local.exists() or path.count("/") != 1:
            # Not a directory, and not shaped like a Hub "org/repo" id.
            raise SystemExit(
                f"{path} is neither a checkpoint directory nor a Hub repo id "
                "(expected 'org/name')."
            )
        entries.append((label, path))
    return entries


def build_batch(tokenizer, args, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fixed batch, shared by every checkpoint so the comparison is paired."""
    dataset = StreamingTokenDataset(
        dataset_name=args.train_dataset,
        tokenizer=tokenizer,
        split=args.train_split,
        config=args.train_config or None,
        max_length=args.max_length,
        max_samples=args.batch_size,
        skip_samples=0,
        shuffle_buffer=0,  # deterministic: always the first N rows
        seed=args.seed,
    )
    rows = list(dataset)
    if len(rows) < args.batch_size:
        raise SystemExit(f"only got {len(rows)} samples, wanted {args.batch_size}")

    input_ids = torch.stack([r[0] for r in rows]).to(device)
    attention_mask = torch.stack([r[1] for r in rows]).to(device)
    labels = torch.stack([r[2] for r in rows]).to(device)
    return input_ids, attention_mask, labels


def print_curves(curves: dict[str, list[dict]], metric: str, n_steps: int) -> None:
    labels = list(curves)
    width = max(9, *(len(x) for x in labels))
    base = labels[0]

    header = f"{'step':>4} " + " ".join(f"{x:>{width}}" for x in labels)
    if len(labels) > 1:
        header += "  |  " + " ".join(f"{'Δ ' + x:>{width}}" for x in labels[1:])
    print(f"\n=== {metric} vs reasoning step ===")
    print(header)
    print("-" * len(header))

    for step in range(n_steps):
        values = {x: curves[x][step][metric] for x in labels}
        line = f"{step:>4} " + " ".join(f"{values[x]:{width}.6f}" for x in labels)
        if len(labels) > 1:
            line += "  |  " + " ".join(
                f"{values[x] - values[base]:+{width}.6f}" for x in labels[1:]
            )
        print(line)

    print()
    for x in labels:
        series = [c[metric] for c in curves[x]]
        lo = min(range(len(series)), key=lambda i: series[i])
        print(f"{x:>{width}}: min={series[lo]:.6f} @ step {lo} | "
              f"first={series[0]:.6f} last={series[-1]:.6f}")

    if len(labels) > 1:
        print("\nmax |Δ| vs " + base + ":")
        for x in labels[1:]:
            diffs = [abs(curves[x][i][metric] - curves[base][i][metric]) for i in range(n_steps)]
            print(f"{x:>{width}}: {max(diffs):.6f} (at step {diffs.index(max(diffs))})")


def write_csv(curves: dict[str, list[dict]], path: str, n_steps: int) -> None:
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "step", *METRICS])
        for label, series in curves.items():
            for step in range(n_steps):
                writer.writerow([label, step, *(series[step][m] for m in METRICS)])
    print(f"\n[csv] wrote {path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    entries = parse_checkpoints(args.checkpoints)
    load_device, _, dtype = resolve_device_and_dtype(args.device)
    accelerator = Accelerator()
    cfg = TrainConfig(max_length=args.max_length, n_reasoning_steps=args.n_reasoning_steps)

    batch = None
    curves: dict[str, list[dict]] = {}

    for label, path in entries:
        print(f"[load] {label} <- {path}")
        model = load_lds_model(
            model_name="",
            device=load_device,
            dtype=dtype,
            decomposed_model=path,
            n_recursion=args.n_reasoning_steps,
        )
        if batch is None:
            batch = build_batch(model.tokenizer, args, accelerator.device)
            print(f"[data] fixed batch: {tuple(batch[0].shape)}")

        input_ids, attention_mask, labels = batch
        curves[label] = _collect_reasoning_step_losses(
            combined_model=model,
            accelerator=accelerator,
            input_ids=input_ids,
            attention_mask=attention_mask.to(dtype=torch.bool),
            labels=labels,
            cfg=cfg,
            n_reasoning_steps=args.n_reasoning_steps,
        )
        del model
        torch.cuda.empty_cache()

    print_curves(curves, args.metric, args.n_reasoning_steps)
    if args.csv:
        write_csv(curves, args.csv, args.n_reasoning_steps)


if __name__ == "__main__":
    main()
