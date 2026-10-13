"""Prepare a local model snapshot and a bounded dataset before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


MANIFEST = "loopus_download.json"


def sample_budget(max_tokens: int, max_samples: int, max_length: int) -> int:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if max_tokens > 0:
        max_samples = max_tokens // max_length
        if max_samples == 0:
            raise ValueError("train_max_tokens must be at least max_length")
    return max(max_samples, 0)


def _ready(path: Path, expected: dict) -> bool:
    manifest = path / MANIFEST
    if not manifest.is_file():
        return False
    actual = json.loads(manifest.read_text())
    if actual != expected:
        raise ValueError(
            f"Download settings differ from {manifest}. Use a new local directory "
            "or move the existing directory before downloading again."
        )
    return True


def prepare_model(model_name: str, output: Path, check_only: bool = False) -> None:
    expected = {"model_name": model_name}
    if _ready(output, expected):
        if not (output / "config.json").is_file():
            raise ValueError(f"Incomplete model at {output}; download into a new directory.")
        print(f"[model] Ready: {output}", flush=True)
        return
    if check_only:
        raise ValueError(f"Model is not ready at {output}. Run bash script/download.sh first.")
    print(f"[model] Downloading {model_name} -> {output}", flush=True)
    snapshot_download(repo_id=model_name, local_dir=str(output))
    (output / MANIFEST).write_text(json.dumps(expected, indent=2) + "\n")


def _dataset_rows(dataset_name: str, config: str, split: str, max_samples: int):
    dataset = load_dataset(dataset_name, name=config or None, split=split, streaming=True)
    if max_samples > 0:
        dataset = dataset.take(max_samples)
    yield from dataset


def prepare_dataset(
    dataset_name: str, config: str, split: str, max_samples: int,
    output: Path, check_only: bool = False,
) -> None:
    expected = {
        "dataset_name": dataset_name, "config": config,
        "split": split, "max_samples": max_samples,
    }
    if _ready(output, expected):
        dataset = load_from_disk(str(output))
        print(f"[dataset] Ready: {output} ({len(dataset):,} rows)", flush=True)
        return
    if check_only:
        raise ValueError(f"Dataset is not ready at {output}. Run bash script/download.sh first.")
    if output.exists():
        raise ValueError(f"Unrecognized dataset directory: {output}. Choose a new directory.")

    print(f"[dataset] Downloading {dataset_name} / {config} / {split} "
          f"({max_samples or 'all'} rows) -> {output}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build on disk, then publish only a fully saved dataset. Interrupted runs
    # never leave a final directory that training could mistake for complete.
    with TemporaryDirectory(prefix=".loopus-download-", dir=output.parent) as temp:
        dataset = Dataset.from_generator(
            _dataset_rows,
            gen_kwargs={
                "dataset_name": dataset_name, "config": config,
                "split": split, "max_samples": max_samples,
            },
            cache_dir=str(Path(temp) / "cache"),
        )
        if len(dataset) == 0:
            raise ValueError("The selected dataset is empty.")
        staged = Path(temp) / "dataset"
        dataset.save_to_disk(str(staged))
        (staged / MANIFEST).write_text(json.dumps(expected, indent=2) + "\n")
        staged.rename(output)
    print(f"[dataset] Saved {len(dataset):,} rows to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--train-config", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-max-tokens", type=int, default=-1)
    parser.add_argument("--train-max-samples", type=int, default=200_000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="Check local assets without downloading anything")
    args = parser.parse_args()
    load_dotenv()
    try:
        budget = sample_budget(args.train_max_tokens, args.train_max_samples, args.max_length)
        prepare_model(args.model_name, args.model_dir, args.check_only)
        prepare_dataset(args.train_dataset, args.train_config, args.train_split,
                        budget, args.dataset_dir, args.check_only)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
