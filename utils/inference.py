"""Shared helpers for loading LDS models in evaluation and sampling scripts."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from models.configuration_lds import LDSConfig
from models.modeling_lds import LDSForCausalLM, _remap_state_dict_keys
from utils.common import parse_layer_indices


def resolve_device_and_dtype(
    requested_device: str = "auto",
) -> tuple[str | torch.device, str | torch.device, torch.dtype]:
    """Resolve model-load device, adapter device, and default dtype."""
    if requested_device == "auto":
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        return "auto", "auto", dtype

    adapter_device = torch.device(requested_device)
    use_cuda = adapter_device.type == "cuda"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    return adapter_device, adapter_device, dtype


def load_lds_model(
    model_name: str,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    encoder_layers: str | list[int] | None = None,
    decoder_layers: str | list[int] | None = None,
    decomposed_model: str | None = None,
    checkpoint_dir: str | None = None,
    n_recursion: int = 1,
    q_stop_threshold: float | None = None,
    q_eval_interval: int = 1,
    halting_strategy: str = "threshold",
    convergence_epsilon: float = 1e-2,
) -> LDSForCausalLM:
    """Load an LDS model from Hub, save_pretrained directory, or legacy checkpoint."""
    resolved_encoder_layers = (
        parse_layer_indices(encoder_layers)
        if isinstance(encoder_layers, str)
        else encoder_layers
    )
    resolved_decoder_layers = (
        parse_layer_indices(decoder_layers)
        if isinstance(decoder_layers, str)
        else decoder_layers
    )
    device_map = str(device)

    if decomposed_model:
        print(f"Loading saved LDS model from: {decomposed_model}")
        combined = LDSForCausalLM.from_pretrained(
            decomposed_model,
            torch_dtype=dtype,
            device_map=device_map,
        )
    else:
        print(f"Loading decomposed model from base: {model_name}")
        config = LDSConfig(
            base_model_name_or_path=model_name,
            encoder_layer_indices=resolved_encoder_layers,
            decoder_layer_indices=resolved_decoder_layers,
            N=n_recursion,
            q_threshold=0.6 if q_stop_threshold is None else q_stop_threshold,
            q_eval_interval=q_eval_interval,
        )
        combined = LDSForCausalLM.from_pretrained(
            config,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map=device_map,
        )

    tokenizer = combined.tokenizer
    if tokenizer is None:
        tokenizer_source = decomposed_model or model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        combined.tokenizer = tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_dir and not decomposed_model:
        ckpt_path = os.path.join(checkpoint_dir, "combined_model.pt")
        if os.path.isfile(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            cleaned = {k.replace(".module.", "."): v for k, v in state_dict.items()}
            cleaned = _remap_state_dict_keys(cleaned)
            combined.load_state_dict(cleaned)
            print(f"LDSForCausalLM weights loaded from {ckpt_path}")
        else:
            print(f"WARNING: No combined_model.pt found in {checkpoint_dir}")

    combined.N = n_recursion
    combined.config.N = n_recursion
    combined.q_eval_interval = max(1, int(q_eval_interval))
    combined.config.q_eval_interval = combined.q_eval_interval
    if q_stop_threshold is not None:
        combined.q_threshold = q_stop_threshold
        combined.config.q_threshold = q_stop_threshold

    combined.halting_strategy = halting_strategy
    combined.config.halting_strategy = halting_strategy
    combined.convergence_epsilon = convergence_epsilon
    combined.config.convergence_epsilon = convergence_epsilon

    combined.eval()
    return combined