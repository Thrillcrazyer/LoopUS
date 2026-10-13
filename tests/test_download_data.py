"""CPU integration tests for download-once/local-training data flows."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset, DatasetDict, load_from_disk

from download import MANIFEST, prepare_dataset, prepare_model, sample_budget
from utils.data import _estimate_split_sizes, create_streaming_dataloaders


class TinyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[int(text)] * kwargs["max_length"]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def tiny_rows(**kwargs):
    for i in range(kwargs["max_samples"] or 20):
        yield {"text": str(i)}


def broken_rows(**kwargs):
    yield {"text": "0"}
    raise RuntimeError("connection interrupted")


class DownloadDataTests(unittest.TestCase):
    def test_saved_data_trains_offline_without_worker_duplicates_or_eval_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dataset"
            with patch("download._dataset_rows", tiny_rows):
                prepare_dataset("test/data", "subset", "train", 20, output)
            with patch("download.load_dataset", side_effect=AssertionError("network")), \
                 patch("utils.data.load_dataset", side_effect=AssertionError("network")):
                prepare_dataset("test/data", "subset", "train", 20, output, check_only=True)
                train, val, n_train, n_val = create_streaming_dataloaders(
                    TinyTokenizer(), str(output), max_samples=100, val_ratio=0.2,
                    max_length=2, batch_size=2, num_workers=2, pin_memory=False,
                    train_alignment_samples=4,
                )
                train_ids = [int(i) for batch in train for i in batch[0][:, 0]]
                val_ids = [int(i) for batch in val for i in batch[0][:, 0]]
            self.assertEqual((n_train, n_val), (16, 4))
            self.assertEqual(sorted(train_ids), list(range(16)))
            self.assertEqual(sorted(val_ids), list(range(16, 20)))

    def test_budget_bounds_download_and_reuses_completed_data(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dataset"
            source = Path(temp) / "source"
            source.mkdir()
            (source / "train.jsonl").write_text(
                "".join(json.dumps({"text": str(i)}) + "\n" for i in range(20))
            )
            prepare_dataset(str(source), "", "train", 7, output)
            self.assertEqual(len(load_from_disk(str(output))), 7)
            with patch("download.load_dataset", side_effect=AssertionError("network")):
                prepare_dataset(str(source), "", "train", 7, output)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                prepare_dataset(str(source), "", "train", 8, output)

    def test_interrupted_download_is_not_ready_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dataset"
            with patch("download._dataset_rows", broken_rows):
                with self.assertRaises(Exception):
                    prepare_dataset("test/data", "", "train", 5, output)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "download.sh first"):
                prepare_dataset("test/data", "", "train", 5, output, check_only=True)
            with patch("download._dataset_rows", tiny_rows):
                prepare_dataset("test/data", "", "train", 5, output)
            self.assertEqual(len(load_from_disk(str(output))), 5)

    def test_model_is_marked_ready_only_after_download(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "model"
            with patch("download.snapshot_download", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    prepare_model("test/model", output)
            self.assertFalse((output / MANIFEST).exists())

            def snapshot(**kwargs):
                output.mkdir()
                (output / "config.json").write_text("{}")

            with patch("download.snapshot_download", side_effect=snapshot):
                prepare_model("test/model", output)
            with patch("download.snapshot_download", side_effect=AssertionError("network")):
                prepare_model("test/model", output, check_only=True)
            self.assertEqual(json.loads((output / MANIFEST).read_text()),
                             {"model_name": "test/model"})

    def test_saved_dataset_dict_uses_requested_split(self):
        with tempfile.TemporaryDirectory() as temp:
            DatasetDict({
                "train_sft": Dataset.from_dict({"messages": [[{"role": "assistant", "content": "hi"}]] * 8}),
                "test": Dataset.from_dict({"text": ["0"]}),
            }).save_to_disk(temp)
            self.assertEqual(_estimate_split_sizes(temp, "train_sft", None, 0, 0.25), (6, 2))

    def test_token_budget_matches_training_and_rejects_zero_row_budget(self):
        self.assertEqual(sample_budget(3_000_000_000, 200_000, 1024), 2_929_687)
        self.assertEqual(sample_budget(-1, 10, 1024), 10)
        self.assertEqual(sample_budget(-1, -1, 1024), 0)
        with self.assertRaises(ValueError):
            sample_budget(100, 10, 1024)

    def test_tiny_local_dataset_cannot_overlap_training_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            Dataset.from_dict({"text": ["0"]}).save_to_disk(temp)
            with self.assertRaisesRegex(ValueError, "no training rows"):
                _estimate_split_sizes(temp, "train", None, 100, 0.1)


if __name__ == "__main__":
    unittest.main()
