"""Streaming data utilities for pretraining and SFT training."""

from __future__ import annotations

import itertools
from typing import Callable

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import PreTrainedTokenizer


def _ensure_pad_token(tokenizer: PreTrainedTokenizer) -> None:
    """Ensure the tokenizer has a pad token for fixed-length batches."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _messages_to_text(messages: list[dict]) -> str:
    """Convert a list of chat-style messages to a single string."""
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip()
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts).strip()


def _example_to_text(example: dict) -> str:
    """Extract a plain-text string from a single dataset example."""
    if "messages" in example and isinstance(example["messages"], list):
        text = _messages_to_text(example["messages"])
        if text:
            return text

    if "conversations" in example and isinstance(example["conversations"], list):
        text = _messages_to_text(example["conversations"])
        if text:
            return text

    for key in ("text", "prompt", "instruction", "response", "output"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _extract_messages(example: dict) -> list[dict] | None:
    """Extract chat messages from a dataset example."""
    for key in ("messages", "conversations"):
        messages = example.get(key)
        if isinstance(messages, list) and messages:
            return messages
    return None


def _tokenize_text(
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize plain text into fixed-length causal LM tensors."""
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def _build_sft_tensors(
    token_ids: list[int],
    label_mask: list[bool],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad token ids and apply assistant-only label masking."""
    seq_len = min(len(token_ids), max_length)
    token_ids = token_ids[:seq_len]
    label_mask = label_mask[:seq_len]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_len = max_length - seq_len

    input_ids = torch.tensor(token_ids + [pad_id] * pad_len, dtype=torch.long)
    attention_mask = torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long)
    labels = torch.full((max_length,), -100, dtype=torch.long)

    for index in range(seq_len):
        if label_mask[index]:
            labels[index] = input_ids[index]

    return input_ids, attention_mask, labels


def _tokenize_sft_chat(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a chat conversation with assistant-only label masking."""
    if not hasattr(tokenizer, "apply_chat_template"):
        return _tokenize_text(tokenizer, _messages_to_text(messages), max_length)

    full_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )

    label_mask = [False] * len(full_ids)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    has_chatml = (
        im_start_id is not None
        and im_start_id != getattr(tokenizer, "unk_token_id", None)
        and im_end_id is not None
        and im_end_id != getattr(tokenizer, "unk_token_id", None)
    )

    if has_chatml:
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_ids[0] if newline_ids else None

        turn_index = 0
        cursor = 0
        while cursor < len(full_ids):
            if full_ids[cursor] == im_start_id:
                end_cursor = cursor + 1
                while end_cursor < len(full_ids) and full_ids[end_cursor] != im_end_id:
                    end_cursor += 1

                if turn_index < len(messages) and messages[turn_index].get("role") == "assistant":
                    header_cursor = cursor + 1
                    if newline_id is not None:
                        while header_cursor < end_cursor and full_ids[header_cursor] != newline_id:
                            header_cursor += 1
                        content_start = header_cursor + 1
                    else:
                        content_start = cursor + 2

                    for token_index in range(content_start, min(end_cursor + 1, len(full_ids))):
                        label_mask[token_index] = True

                turn_index += 1
                cursor = end_cursor + 1
            else:
                cursor += 1
    else:
        label_mask = [True] * len(full_ids)

    return _build_sft_tensors(full_ids, label_mask, tokenizer, max_length)


class _BaseStreamingDataset(IterableDataset):
    """Shared Hugging Face streaming wrapper for tokenized iterable datasets."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        split: str,
        config: str | None,
        max_length: int,
        max_samples: int,
        skip_samples: int,
        shuffle_buffer: int,
        seed: int,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.split = split
        self.config = config
        self.max_length = max_length
        self.max_samples = max_samples
        self.skip_samples = skip_samples
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        _ensure_pad_token(tokenizer)

    def _open_stream(self):
        load_kwargs: dict = {"split": self.split, "streaming": True}
        if self.config:
            load_kwargs["name"] = self.config
        dataset = load_dataset(self.dataset_name, **load_kwargs)

        if self.skip_samples > 0:
            dataset = dataset.skip(self.skip_samples)
        if self.max_samples and self.max_samples > 0:
            dataset = dataset.take(self.max_samples)
        if self.shuffle_buffer and self.shuffle_buffer > 1:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        return dataset

    def _iter_stream(self):
        dataset = self._open_stream()
        worker = get_worker_info()
        if worker is None:
            return iter(dataset)
        return itertools.islice(dataset, worker.id, None, worker.num_workers)

    def _convert_example(
        self,
        example: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError

    def __iter__(self):
        for example in self._iter_stream():
            converted = self._convert_example(example)
            if converted is not None:
                yield converted


class StreamingTokenDataset(_BaseStreamingDataset):
    """Tokenize plain-text examples on the fly from a streaming dataset."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        config: str | None = None,
        max_length: int = 2048,
        max_samples: int = 0,
        skip_samples: int = 0,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ):
        super().__init__(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=split,
            config=config,
            max_length=max_length,
            max_samples=max_samples,
            skip_samples=skip_samples,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
        )

    def _convert_example(
        self,
        example: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        text = _example_to_text(example)
        if not text:
            return None
        return _tokenize_text(self.tokenizer, text, self.max_length)


class StreamingSFTDataset(_BaseStreamingDataset):
    """Streaming SFT dataset with chat-template formatting and assistant-only masking."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train_sft",
        config: str | None = None,
        max_length: int = 2048,
        max_samples: int = 0,
        skip_samples: int = 0,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ):
        super().__init__(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=split,
            config=config,
            max_length=max_length,
            max_samples=max_samples,
            skip_samples=skip_samples,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
        )

    def _convert_example(
        self,
        example: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        messages = _extract_messages(example)
        if not messages:
            return None
        return _tokenize_sft_chat(messages, self.tokenizer, self.max_length)


def _estimate_split_sizes(
    dataset_name: str,
    split: str,
    config: str | None,
    max_samples: int,
    val_ratio: float,
    train_alignment_samples: int = 1,
) -> tuple[int, int]:
    """Estimate train/eval sizes from an explicit budget or dataset metadata."""
    if max_samples and max_samples > 0:
        n_eval = max(1, int(max_samples * val_ratio)) if val_ratio > 0 else 0
        n_train = max_samples - n_eval
        if train_alignment_samples > 1 and n_train >= train_alignment_samples:
            n_train = (n_train // train_alignment_samples) * train_alignment_samples
        return n_train, n_eval

    try:
        probe_kwargs: dict = {"split": split, "streaming": True}
        if config:
            probe_kwargs["name"] = config
        probe_ds = load_dataset(dataset_name, **probe_kwargs)
        total = probe_ds.info.splits[split].num_examples
        if total and total > 0:
            n_eval = max(1, int(total * val_ratio)) if val_ratio > 0 else 0
            return total - n_eval, n_eval
    except Exception:
        pass

    return 0, 0


def _create_streaming_dataloaders(
    dataset_factory: Callable[..., IterableDataset],
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    split: str,
    config: str | None,
    max_samples: int,
    val_ratio: float,
    max_length: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle_buffer: int,
    seed: int,
    train_alignment_samples: int = 1,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    """Create train/eval dataloaders for a streaming dataset factory."""
    n_train, n_eval = _estimate_split_sizes(
        dataset_name=dataset_name,
        split=split,
        config=config,
        max_samples=max_samples,
        val_ratio=val_ratio,
        train_alignment_samples=train_alignment_samples,
    )

    train_ds = dataset_factory(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split=split,
        config=config,
        max_length=max_length,
        max_samples=n_train,
        skip_samples=0,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    eval_dataloader = None
    if val_ratio > 0 and n_eval > 0:
        eval_ds = dataset_factory(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=split,
            config=config,
            max_length=max_length,
            max_samples=n_eval,
            skip_samples=n_train,
            shuffle_buffer=0,
            seed=seed,
        )
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_dataloader, eval_dataloader, n_train, n_eval


def create_sft_streaming_dataloaders(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    split: str = "train_sft",
    config: str | None = None,
    max_samples: int = 200_000,
    val_ratio: float = 0.01,
    max_length: int = 2048,
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
    train_alignment_samples: int = 1,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    """Create streaming SFT train/eval DataLoaders with assistant-only masking."""
    return _create_streaming_dataloaders(
        dataset_factory=StreamingSFTDataset,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        config=config,
        max_samples=max_samples,
        val_ratio=val_ratio,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        train_alignment_samples=train_alignment_samples,
    )


def create_streaming_dataloaders(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    split: str = "train",
    config: str | None = None,
    max_samples: int = 200_000,
    val_ratio: float = 0.01,
    max_length: int = 2048,
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
    train_alignment_samples: int = 1,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    """Create streaming train/eval DataLoaders for plain-text datasets."""
    return _create_streaming_dataloaders(
        dataset_factory=StreamingTokenDataset,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        config=config,
        max_samples=max_samples,
        val_ratio=val_ratio,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        train_alignment_samples=train_alignment_samples,
    )