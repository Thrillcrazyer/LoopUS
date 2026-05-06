"""lm-evaluation-harness adapter for LDSForCausalLM (decomposed DDD model).

Provides ``DDDModelAdapter`` — a subclass of lm_eval's ``HFLM`` that delegates
``_model_call`` and ``_model_generate`` to ``LDSForCausalLM.forward`` /
``LDSForCausalLM.generate``, while reusing all of HFLM's tokenisation,
batching, and log-likelihood infrastructure.

Also exposes a thin ``run_lm_eval`` helper that calls ``lm_eval.simple_evaluate``
and returns structured results, used by both ``evaluate.py`` and ``train.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, cast
from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn.functional as F
import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from tqdm.auto import tqdm

from models.modeling_lds import LDSForCausalLM


# ──────────────────────────────────────────────────────────────────────
# Adapter – wraps LDSForCausalLM for lm-evaluation-harness
# ──────────────────────────────────────────────────────────────────────


class DDDModelAdapter(LM):
    """lm-eval adapter for the DDD ``LDSForCausalLM``.

    Implements the three abstract methods required by ``lm_eval.api.model.LM``:
      - ``loglikelihood``
      - ``loglikelihood_rolling``
      - ``generate_until``

    Internally all computation goes through ``LDSForCausalLM.forward`` (which
    runs encoder → reasoning recursion → decoder) and the simple greedy
    ``LDSForCausalLM.generate`` helper.
    """

    def __init__(
        self,
        combined_model: LDSForCausalLM,
        batch_size: int = 1,
        max_length: int = 1024,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.model = combined_model
        self.model.eval()
        if combined_model.tokenizer is None:
            raise ValueError("DDDModelAdapter requires combined_model.tokenizer to be set")
        self.tokenizer: Any = combined_model.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._batch_size = int(batch_size)
        self._max_length = max_length
        self._encode_cache: dict[str, tuple[int, ...]] = {}

        if device is None or str(device) == "auto":
            # Infer from the model's embedding layer — this is where inputs land first
            inferred_device = combined_model.encoder.embed_tokens.weight.device
        else:
            inferred_device = device
        self._device = torch.device(str(inferred_device))

    # ── properties expected by the harness ────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    # ── tokenisation helpers ──────────────────────────────────────────

    def _encode_text(self, text: str) -> tuple[int, ...]:
        cached = self._encode_cache.get(text)
        if cached is None:
            cached = tuple(self.tokenizer.encode(text, add_special_tokens=False))
            self._encode_cache[text] = cached
        return cached

    def _left_pad_batch(
        self,
        sequences: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(ids) for ids in sequences)
        batch_size = len(sequences)
        input_ids = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=self._device,
        )

        for row, ids in enumerate(sequences):
            seq_len = len(ids)
            if seq_len == 0:
                continue
            row_ids = torch.as_tensor(ids, dtype=torch.long, device=self._device)
            input_ids[row, max_len - seq_len :] = row_ids
            attention_mask[row, max_len - seq_len :] = 1

        return input_ids, attention_mask

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return list(self._encode_text(string))

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _encode_pair(self, context: str, continuation: str):
        """Encode a (context, continuation) pair and return token id lists."""
        ctx_enc = self._encode_text(context)
        cont_enc = self._encode_text(continuation)
        return ctx_enc, cont_enc

    # ── model call helpers ────────────────────────────────────────────

    @torch.inference_mode()
    def _model_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run LDSForCausalLM.forward and return logits ``(B, L, V)``."""
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits

    def _batch_starts(self, total_items: int, desc: str, disable_tqdm: bool):
        total_batches = (total_items + self._batch_size - 1) // self._batch_size
        return tqdm(
            range(0, total_items, self._batch_size),
            total=total_batches,
            desc=desc,
            disable=disable_tqdm or total_batches <= 1,
        )

    # ── abstract method implementations ───────────────────────────────

    def loglikelihood(self, requests: list[Instance], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        """Score (context, continuation) pairs.

        Returns list of ``(logprob_sum, is_greedy)`` tuples.
        """
        results: list[tuple[float, bool]] = []

        # Process in batches for efficiency
        for i in self._batch_starts(len(requests), "DDD loglikelihood", disable_tqdm):
            batch = requests[i : i + self._batch_size]

            # Encode each request
            all_input_ids = []
            cont_lengths = []
            for req in batch:
                ctx, cont = cast(tuple[str, str], req.args)
                ctx_ids = self._encode_text(ctx) if ctx else ()
                cont_ids = self._encode_text(cont)
                all_ids = ctx_ids + cont_ids

                # Truncate from the left if needed
                if len(all_ids) > self._max_length:
                    all_ids = all_ids[-self._max_length:]
                    # Recalculate cont_length after truncation
                    cont_lengths.append(min(len(cont_ids), len(all_ids)))
                else:
                    cont_lengths.append(len(cont_ids))

                all_input_ids.append(all_ids)

            input_ids, attention_mask = self._left_pad_batch(all_input_ids)
            max_len = input_ids.shape[1]

            logits = self._model_forward(input_ids, attention_mask)

            for j in range(len(batch)):
                cont_len = cont_lengths[j]
                seq_len = len(all_input_ids[j])
                pad_offset = max_len - seq_len

                # Only extract the positions needed for this continuation
                logit_start = pad_offset + seq_len - cont_len - 1
                logit_end = pad_offset + seq_len - 1
                relevant_logits = logits[j, logit_start:logit_end]  # (cont_len, V)
                relevant_log_probs = F.log_softmax(relevant_logits.float(), dim=-1)

                target_ids = input_ids[j, logit_start + 1 : logit_end + 1]  # (cont_len,)
                token_log_probs = relevant_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                cont_logprob = token_log_probs.sum().item()

                greedy_ids = relevant_logits.argmax(dim=-1)
                is_greedy = bool((greedy_ids == target_ids).all().item())

                results.append((cont_logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:
        """Compute rolling log-likelihood for perplexity.

        Uses sliding window approach matching the standard protocol.
        Batches chunks across requests for efficiency.
        Returns ``list[float]`` (one total log-prob per request) as required
        by the ``lm_eval.api.model.LM`` base class.
        """
        # Pre-compute all chunks: (request_index, chunk_token_ids, target_start)
        all_chunks: list[tuple[int, list[int], int]] = []

        for req_idx, req in enumerate(requests):
            (text,) = cast(tuple[str], req.args)
            token_ids = self._encode_text(text)
            total_len = len(token_ids)

            if total_len == 0:
                continue

            stride = self._max_length // 2 or 1
            for begin in range(0, total_len, stride):
                end = min(begin + self._max_length, total_len)
                chunk_ids = token_ids[begin:end]
                target_start = stride if begin > 0 else 1
                all_chunks.append((req_idx, chunk_ids, target_start))
                if end >= total_len:
                    break

        # Process chunks in batches
        request_logprobs: dict[int, float] = defaultdict(float)

        for i in self._batch_starts(len(all_chunks), "DDD rolling loglikelihood", disable_tqdm):
            batch = all_chunks[i : i + self._batch_size]

            input_ids, attention_mask = self._left_pad_batch([chunk_ids for _, chunk_ids, _ in batch])
            max_len = input_ids.shape[1]

            logits = self._model_forward(input_ids, attention_mask)

            for j, (req_idx, chunk_ids, target_start) in enumerate(batch):
                chunk_len = len(chunk_ids)
                pad_offset = max_len - chunk_len

                logit_start = pad_offset + target_start - 1
                logit_end = pad_offset + chunk_len - 1

                if logit_start >= logit_end:
                    continue

                relevant_logits = logits[j, logit_start:logit_end]
                relevant_log_probs = F.log_softmax(relevant_logits.float(), dim=-1)
                target_ids = input_ids[j, pad_offset + target_start : pad_offset + chunk_len]
                request_logprobs[req_idx] += (
                    relevant_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).sum().item()
                )

        return [request_logprobs.get(i, 0.0) for i in range(len(requests))]

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        """Generate text until a stop condition is met."""
        results: list[str | None] = [None] * len(requests)
        grouped: dict[tuple[tuple[str, ...], int, float, bool], list[tuple[int, str]]] = defaultdict(list)

        for index, req in enumerate(requests):
            context, gen_kwargs = cast(tuple[str, dict[str, Any]], req.args)
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = gen_kwargs.get("do_sample", False)
            key = (tuple(until), int(max_gen_toks), float(temperature), bool(do_sample))
            grouped[key].append((index, context))

        total_generate_batches = sum(
            (len(entries) + self._batch_size - 1) // self._batch_size
            for entries in grouped.values()
        )
        generate_pbar = tqdm(
            total=total_generate_batches,
            desc="DDD generate_until",
            disable=disable_tqdm or total_generate_batches <= 1,
        )

        for (until, max_gen_toks, temperature, do_sample), entries in grouped.items():
            for start in range(0, len(entries), self._batch_size):
                batch_entries = entries[start : start + self._batch_size]
                batch_indices = [item[0] for item in batch_entries]

                encoded_contexts = []
                for _, context in batch_entries:
                    ctx_ids = self._encode_text(context)
                    max_ctx_len = max(1, self._max_length - max_gen_toks)
                    if len(ctx_ids) > max_ctx_len:
                        ctx_ids = ctx_ids[-max_ctx_len:]
                    encoded_contexts.append(ctx_ids)

                input_ids, attention_mask = self._left_pad_batch(encoded_contexts)

                generated = self.model.generate(
                    input_ids=cast(torch.LongTensor, input_ids.long()),
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen_toks,
                    do_sample=do_sample or (temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_context=self._max_length,
                )

                gen_token_ids = generated[:, input_ids.shape[1]:]
                decoded_texts = self.tokenizer.batch_decode(gen_token_ids, skip_special_tokens=True)
                for row, request_index in enumerate(batch_indices):
                    gen_text = decoded_texts[row]
                    for stop_seq in until:
                        idx = gen_text.find(stop_seq)
                        if idx != -1:
                            gen_text = gen_text[:idx]
                    results[request_index] = gen_text

                generate_pbar.update(1)

        generate_pbar.close()

        return [text if text is not None else "" for text in results]


# ──────────────────────────────────────────────────────────────────────
# Adapter for vanilla AutoModelForCausalLM (original / non-decomposed)
# ──────────────────────────────────────────────────────────────────────


class OriginalModelAdapter(LM):
    """lm-eval adapter for a vanilla ``AutoModelForCausalLM``.

    Same interface as ``DDDModelAdapter`` but delegates to the standard HF model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        batch_size: int = 1,
        max_length: int = 2048,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        if tokenizer is None:
            raise ValueError("OriginalModelAdapter requires a tokenizer")
        self.tokenizer: Any = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._batch_size = int(batch_size)
        self._max_length = max_length
        self._uses_auto_device_map = hasattr(model, "hf_device_map") and bool(model.hf_device_map)

        if device == "auto":
            device = None

        if device is None:
            device = self._infer_input_device()
        self._device = torch.device(device)

    def _infer_input_device(self) -> torch.device:
        if self._uses_auto_device_map:
            for mapped_device in self.model.hf_device_map.values():
                if isinstance(mapped_device, int):
                    return torch.device(f"cuda:{mapped_device}")
                if isinstance(mapped_device, str) and mapped_device.startswith("cuda"):
                    return torch.device(mapped_device)

        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return torch.device(model_device)

        return next(self.model.parameters()).device

    def _batch_starts(self, total_items: int, desc: str, disable_tqdm: bool):
        total_batches = (total_items + self._batch_size - 1) // self._batch_size
        return tqdm(
            range(0, total_items, self._batch_size),
            total=total_batches,
            desc=desc,
            disable=disable_tqdm or total_batches <= 1,
        )

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def _model_forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def loglikelihood(self, requests: list[Instance], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for i in self._batch_starts(len(requests), "HF loglikelihood", disable_tqdm):
            batch = requests[i : i + self._batch_size]
            all_input_ids = []
            cont_lengths = []
            for req in batch:
                ctx, cont = cast(tuple[str, str], req.args)
                ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=False) if ctx else []
                cont_ids = self.tokenizer.encode(cont, add_special_tokens=False)
                all_ids = ctx_ids + cont_ids
                if len(all_ids) > self._max_length:
                    all_ids = all_ids[-self._max_length:]
                    cont_lengths.append(min(len(cont_ids), len(all_ids)))
                else:
                    cont_lengths.append(len(cont_ids))
                all_input_ids.append(all_ids)

            max_len = max(len(ids) for ids in all_input_ids)
            padded_ids = []
            attention_masks = []
            for ids in all_input_ids:
                pad_len = max_len - len(ids)
                padded_ids.append([self.tokenizer.pad_token_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))

            input_ids = torch.tensor(padded_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)
            logits = self._model_forward(input_ids, attention_mask)
            logits_device = logits.device

            for j in range(len(batch)):
                cont_len = cont_lengths[j]
                seq_len = len(all_input_ids[j])
                pad_offset = max_len - seq_len

                logit_start = pad_offset + seq_len - cont_len - 1
                logit_end = pad_offset + seq_len - 1
                relevant_logits = logits[j, logit_start:logit_end]
                relevant_log_probs = F.log_softmax(relevant_logits.float(), dim=-1)

                target_ids = input_ids[j, logit_start + 1 : logit_end + 1].to(logits_device)
                token_log_probs = relevant_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                cont_logprob = token_log_probs.sum().item()

                greedy_ids = relevant_logits.argmax(dim=-1)
                is_greedy = bool((greedy_ids == target_ids).all().item())
                results.append((cont_logprob, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:
        results: list[float] = []
        request_iterator = tqdm(
            requests,
            total=len(requests),
            desc="HF rolling loglikelihood",
            disable=disable_tqdm or len(requests) <= 1,
        )
        for req in request_iterator:
            (text,) = cast(tuple[str], req.args)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            total_len = len(token_ids)
            if total_len == 0:
                results.append(0.0)
                continue
            total_logprob = 0.0
            stride = self._max_length // 2 or 1
            for begin in range(0, total_len, stride):
                end = min(begin + self._max_length, total_len)
                chunk_ids = token_ids[begin:end]
                input_ids = torch.tensor([chunk_ids], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                logits = self._model_forward(input_ids, attention_mask)
                target_start = max(1, begin - (begin - stride) if begin > 0 else 1)
                relevant_logits = logits[0, target_start - 1 : len(chunk_ids) - 1]
                relevant_log_probs = F.log_softmax(relevant_logits.float(), dim=-1)
                target_ids = torch.tensor(chunk_ids[target_start:], dtype=torch.long, device=logits.device)
                total_logprob += relevant_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).sum().item()
                if end >= total_len:
                    break
            results.append(total_logprob)
        return results

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        results: list[str | None] = [None] * len(requests)
        grouped: dict[tuple[tuple[str, ...], int, float, bool], list[tuple[int, str]]] = defaultdict(list)

        for index, req in enumerate(requests):
            context, gen_kwargs = cast(tuple[str, dict[str, Any]], req.args)
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = gen_kwargs.get("do_sample", False)
            key = (tuple(until), int(max_gen_toks), float(temperature), bool(do_sample))
            grouped[key].append((index, context))

        total_generate_batches = sum(
            (len(entries) + self._batch_size - 1) // self._batch_size
            for entries in grouped.values()
        )
        generate_pbar = tqdm(
            total=total_generate_batches,
            desc="HF generate_until",
            disable=disable_tqdm or total_generate_batches <= 1,
        )

        for (until, max_gen_toks, temperature, do_sample), entries in grouped.items():
            for start in range(0, len(entries), self._batch_size):
                batch_entries = entries[start : start + self._batch_size]
                batch_indices = [item[0] for item in batch_entries]

                encoded_contexts = []
                for _, context in batch_entries:
                    ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
                    max_ctx_len = max(1, self._max_length - max_gen_toks)
                    if len(ctx_ids) > max_ctx_len:
                        ctx_ids = ctx_ids[-max_ctx_len:]
                    encoded_contexts.append(ctx_ids)

                max_ctx_len = max(len(ids) for ids in encoded_contexts)
                padded_ids = []
                attention_masks = []
                for ids in encoded_contexts:
                    pad_len = max_ctx_len - len(ids)
                    padded_ids.append([self.tokenizer.pad_token_id] * pad_len + ids)
                    attention_masks.append([0] * pad_len + [1] * len(ids))

                input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self._device)
                attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self._device)

                gen_kwargs_hf = {
                    "max_new_tokens": max_gen_toks,
                    "do_sample": do_sample or (temperature > 0),
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_kwargs_hf["temperature"] = temperature

                try:
                    output = self.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs_hf)
                    gen_token_ids = output[:, input_ids.shape[1]:]
                except Exception:
                    generated = input_ids.clone()
                    generated_mask = attention_mask.clone()
                    for _ in range(max_gen_toks):
                        model_input = generated[:, -self._max_length:] if generated.shape[1] > self._max_length else generated
                        model_mask = generated_mask[:, -model_input.shape[1]:]
                        logits = self._model_forward(model_input, model_mask)
                        next_logits = logits[:, -1, :]
                        if temperature > 0 and do_sample:
                            probs = F.softmax(next_logits / temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = next_logits.argmax(dim=-1, keepdim=True)
                        next_token = next_token.to(generated.device)
                        generated = torch.cat([generated, next_token], dim=-1)
                        generated_mask = torch.cat(
                            [generated_mask, torch.ones((generated_mask.shape[0], 1), dtype=generated_mask.dtype, device=generated_mask.device)],
                            dim=-1,
                        )
                        if (next_token == self.tokenizer.eos_token_id).all():
                            break
                    gen_token_ids = generated[:, input_ids.shape[1]:]

                for row, request_index in enumerate(batch_indices):
                    gen_text = self.tokenizer.decode(gen_token_ids[row].tolist(), skip_special_tokens=True)
                    for stop_seq in until:
                        idx = gen_text.find(stop_seq)
                        if idx != -1:
                            gen_text = gen_text[:idx]
                    results[request_index] = gen_text

                generate_pbar.update(1)

        generate_pbar.close()

        return [text if text is not None else "" for text in results]
        return results


# ──────────────────────────────────────────────────────────────────────
# High-level evaluation driver
# ──────────────────────────────────────────────────────────────────────

# Default benchmark suite — replaces custom MMLU + WikiText-PPL
DEFAULT_TASKS = [
    "mmlu",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "lambada_openai",
    "wikitext",
    "boolq",
    "openbookqa",
]


def _extract_results(raw_results: dict) -> dict:
    """Flatten lm_eval results into a simple dict for logging.

    Returns a dict like::

        {
            "mmlu": {"acc": 0.25, "acc_norm": 0.26, ...},
            "hellaswag": {"acc": 0.31, "acc_norm": 0.34},
            ...
            "summary": {"avg_acc_norm": 0.30, "n_tasks": 10, ...}
        }
    """
    task_results: dict[str, dict[str, Any]] = {}
    acc_norms: list[float] = []
    accs: list[float] = []

    if "results" not in raw_results:
        return {"summary": {"error": "no results key in output"}}

    for task_name, metrics in raw_results["results"].items():
        clean: dict[str, Any] = {}
        for k, v in metrics.items():
            # Skip metadata keys
            if k.startswith("alias"):
                continue
            # lm_eval encodes metric names like "acc,none" or "acc_norm,none"
            clean_key = k.split(",")[0] if "," in k else k
            # Avoid duplicates — keep first occurrence
            if clean_key not in clean:
                clean[clean_key] = v
        task_results[task_name] = clean

        # Collect accuracy values for summary
        if "acc_norm" in clean and clean["acc_norm"] is not None:
            acc_norms.append(float(clean["acc_norm"]))
        elif "acc" in clean and clean["acc"] is not None:
            accs.append(float(clean["acc"]))

        # For perplexity tasks like wikitext
        if "word_perplexity" in clean:
            pass  # include it in per-task results

    # Build summary
    all_accs = acc_norms + accs
    summary: dict[str, Any] = {
        "n_tasks": len(task_results),
    }
    if all_accs:
        summary["avg_acc"] = sum(all_accs) / len(all_accs)
    if acc_norms:
        summary["avg_acc_norm"] = sum(acc_norms) / len(acc_norms)

    task_results["summary"] = summary
    return task_results


def run_lm_eval(
    model: LM,
    tasks: list[str] | None = None,
    num_fewshot: int | None = None,
    batch_size: int | None = None,
    limit: int | float | None = None,
    log_samples: bool = False,
    verbosity: str | None = "INFO",
    device: str | torch.device | None = None,
    bootstrap_iters: int = 0,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
) -> dict:
    """Run lm-evaluation-harness and return structured results.

    This is the main entry point used by both ``evaluate.py`` and ``train.py``.

    Args:
        model: An ``LM`` adapter instance (``DDDModelAdapter`` or
            ``OriginalModelAdapter``).
        tasks: List of task names. Defaults to ``DEFAULT_TASKS``.
        num_fewshot: Number of few-shot examples. ``None`` uses task defaults.
        batch_size: Override adapter batch size for the harness.
        limit: Limit samples per task (useful for quick checks during training).
        log_samples: Whether to log individual sample results.
        verbosity: Logging verbosity ("DEBUG", "INFO", "WARNING", "ERROR").
        bootstrap_iters: Number of bootstrap iterations for stderr estimation.
            Use 0 to skip expensive confidence interval computation.
        cache_requests: Cache built lm-eval requests for repeated runs.
        rewrite_requests_cache: Refresh an existing lm-eval request cache.

    Returns:
        A dict with per-task metrics and a ``"summary"`` key.
    """
    if tasks is None:
        tasks = list(DEFAULT_TASKS)

    kwargs: dict[str, Any] = {
        "model": model,
        "tasks": tasks,
        "log_samples": log_samples,
        "device": "auto",
        "bootstrap_iters": int(bootstrap_iters),
        "cache_requests": cache_requests,
        "rewrite_requests_cache": rewrite_requests_cache,
    }
    if num_fewshot is not None:
        kwargs["num_fewshot"] = num_fewshot
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if limit is not None:
        kwargs["limit"] = limit
    if verbosity is not None:
        kwargs["verbosity"] = verbosity

    raw = lm_eval.simple_evaluate(**kwargs)
    if raw is None:
        return {"summary": {"error": "lm_eval.simple_evaluate returned no results"}}
    return _extract_results(raw)


def run_lm_eval_for_training(
    combined_model: LDSForCausalLM,
    tasks: list[str] | None = None,
    batch_size: int = 4,
    max_length: int = 2048,
    limit: int | float | None = None,
    num_fewshot: int | None = None,
    device: str | torch.device | None = None,
    bootstrap_iters: int = 0,
) -> dict:
    """Convenience wrapper for checkpoint-time evaluation during training.

    Creates a temporary ``DDDModelAdapter``, runs the evaluation, and
    returns structured results.  The model is *not* modified.
    """
    adapter = DDDModelAdapter(
        combined_model=combined_model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    return run_lm_eval(
        model=adapter,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        device=device,
        bootstrap_iters=bootstrap_iters,
    )


def format_results_log(results: dict, prefix: str = "") -> str:
    """Format evaluation results into a human-readable log string."""
    lines: list[str] = []
    summary = results.get("summary", {})

    if "avg_acc" in summary:
        lines.append(f"{prefix}avg_acc={summary['avg_acc']:.4f}")
    if "avg_acc_norm" in summary:
        lines.append(f"{prefix}avg_acc_norm={summary['avg_acc_norm']:.4f}")

    for task_name, metrics in sorted(results.items()):
        if task_name == "summary":
            continue
        parts = []
        for k in ["acc", "acc_norm", "word_perplexity", "byte_perplexity", "bits_per_byte"]:
            if k in metrics and metrics[k] is not None:
                if "perplexity" in k:
                    parts.append(f"{k}={float(metrics[k]):.2f}")
                else:
                    parts.append(f"{k}={float(metrics[k]):.4f}")
        if parts:
            lines.append(f"  {task_name}: {', '.join(parts)}")

    return "\n".join(lines)
