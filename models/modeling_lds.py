import json
import math
import os
import shutil
from collections import Counter
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, cast
from pathlib import Path
from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save
from transformers.cache_utils import Cache, DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin, GenerationConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import ModelOutput

from .configuration_lds import LDSConfig


# ------------------------------------------------------------------
# State-dict key remapping (backward compatibility)
# ------------------------------------------------------------------

_KEY_REMAP_RULES: list[tuple[str, str]] = [
    # Old "plateau" prefix → current "reasoning"
    ("plateau.", "reasoning."),
]


def _remap_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap legacy checkpoint key prefixes to current names."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _KEY_REMAP_RULES:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
        out[new_key] = value
    return out

def _normalize_attention_mask(
    attention_mask: Optional[torch.Tensor],
    hidden_states: Optional[torch.Tensor] = None,
    *,
    target_device: Optional[torch.device] = None,
    target_dtype: Optional[torch.dtype] = None,
    causal_mask_cache: Optional[dict[tuple[str, torch.dtype], torch.Tensor]] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Optional[torch.Tensor]:
    """Convert a 2-D padding mask ``(B, T)`` into a 4-D causal + padding
    mask ``(B, 1, T, T)`` suitable for SDPA attention.

    When a 4-D mask is provided to the SDPA kernel the ``is_causal`` flag
    is disabled, so we must embed the causal constraint *inside* the mask
    ourselves.  The previous implementation only broadcast the padding
    mask to ``(B, 1, 1, T)`` which allowed every query position to attend
    to *all* non-padded positions — including future tokens — causing
    severe data leakage and artificially high accuracy.
    """
    if attention_mask is None:
        return None

    if hidden_states is not None:
        target_device = hidden_states.device
        target_dtype = hidden_states.dtype
    elif target_device is None or target_dtype is None:
        raise ValueError("target_device and target_dtype are required when hidden_states is None")

    if attention_mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        attention_mask = attention_mask != 0

    if attention_mask.dim() == 2:
        _, key_value_length = attention_mask.shape
        query_length = hidden_states.shape[1] if hidden_states is not None else len(cache_position)
        # Use the smallest representable value for the dtype instead of
        # -inf to avoid NaN when the softmax input is all -inf.
        min_val = torch.finfo(target_dtype).min
        device_key = str(target_device)
        cache_key = (device_key, target_dtype)

        if cache_position is None:
            cache_position = torch.arange(query_length, device=target_device, dtype=torch.long)
        else:
            cache_position = cache_position.to(device=target_device)

        # Causal mask: allow a query token to attend only up to its absolute cache position.
        cached_causal_mask = None
        if causal_mask_cache is not None:
            cached_causal_mask = causal_mask_cache.get(cache_key)

        required_key_length = max(key_value_length, int(cache_position.max().item()) + 1 if cache_position.numel() > 0 else query_length)
        if cached_causal_mask is None or cached_causal_mask.shape[-1] < required_key_length or cached_causal_mask.shape[-2] < query_length:
            key_positions = torch.arange(required_key_length, device=target_device)
            causal_mask = torch.where(
                key_positions.unsqueeze(0) > cache_position.unsqueeze(1),
                torch.full((query_length, required_key_length), min_val, device=target_device, dtype=target_dtype),
                torch.zeros((query_length, required_key_length), device=target_device, dtype=target_dtype),
            )
            cached_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            if causal_mask_cache is not None:
                causal_mask_cache[cache_key] = cached_causal_mask

        causal_mask = cached_causal_mask[:, :, :query_length, :key_value_length]

        # Padding mask: 0 where valid, min_val where padded  (B, 1, 1, T)
        padding_mask = attention_mask.to(device=target_device, dtype=target_dtype)
        padding_mask = (1.0 - padding_mask) * min_val
        padding_mask = padding_mask[:, None, None, :]      # (B, 1, 1, T)

        # Combined (broadcasts to (B, 1, T, T))
        attention_mask = causal_mask + padding_mask
    
    return attention_mask.to(target_device)


def _resolve_submodule(module: nn.Module, names: tuple[str, ...], *, required: bool = True):
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value

    if required:
        module_name = type(module).__name__
        candidates = ", ".join(names)
        raise AttributeError(f"'{module_name}' is missing expected submodule; tried: {candidates}")

    return None


def _unwrap_layer_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _first_parameter_dtype(*modules: Optional[nn.Module]) -> Optional[torch.dtype]:
    for module in modules:
        if module is None:
            continue
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            continue
    return None


def _crop_cache_object(cache_obj: Any, max_length: int):
    if cache_obj is None or max_length < 0:
        return cache_obj
    if hasattr(cache_obj, "crop"):
        cache_obj.crop(max_length)
    return cache_obj


def _build_attention_mask_mapping(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    cache_position: Optional[torch.LongTensor],
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.LongTensor],
):
    if attention_mask is None:
        return {"full_attention": None, "sliding_attention": None}

    attention_mask = attention_mask.contiguous()

    if attention_mask.dim() == 4:
        return {"full_attention": attention_mask, "sliding_attention": attention_mask}

    if cache_position is None:
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)

    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    full_attention_mask = create_causal_mask(**mask_kwargs)
    sliding_window = getattr(config, "sliding_window", None) or getattr(config, "attention_chunk_size", None)
    if sliding_window is not None:
        sliding_attention_mask = create_sliding_window_causal_mask(**mask_kwargs)
    else:
        sliding_attention_mask = full_attention_mask

    if full_attention_mask is not None:
        full_attention_mask = full_attention_mask.contiguous()
    if sliding_attention_mask is not None:
        sliding_attention_mask = sliding_attention_mask.contiguous()

    return {
        "full_attention": full_attention_mask,
        "sliding_attention": sliding_attention_mask,
    }


@dataclass
class LDSCausalLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[dict[str, Any]] = None

class EncoderBlock(nn.Module):
    """Encoder Block: embed_tokens + rotary_emb + configurable transformer layers.

    Args:
        config: The base model's ``PretrainedConfig``.
        layer_indices: List of layer indices included in this block.
        embed_tokens: The embedding module from the base model.
        rotary_emb: The rotary-embedding module from the base model.
        layers: Pre-extracted transformer layers (in order).
    """
    def __init__(self, config, layer_indices: list[int],
                 embed_tokens: nn.Module, rotary_emb: nn.Module,
                 embed_dropout: Optional[nn.Module],
                 layers: list[nn.Module]):
        super().__init__()
        self.layer_indices = sorted(layer_indices)
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.embed_dropout = embed_dropout
        self.layers = nn.ModuleList(layers)
        self.config = config
        self._cache_position_cache: dict[str, torch.Tensor] = {}

    def _get_cache_position(self, seq_len: int, device: torch.device) -> torch.LongTensor:
        cache_key = str(device)
        cached = self._cache_position_cache.get(cache_key)
        if cached is None or cached.numel() < seq_len:
            cached = torch.arange(seq_len, device=device, dtype=torch.long)
            self._cache_position_cache[cache_key] = cached
        return cast(torch.LongTensor, cached[:seq_len])
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        # Move inputs to the same device as embed_tokens weights
        embed_device = cast(torch.device, self.embed_tokens.weight.device)
        input_ids = cast(torch.LongTensor, input_ids.to(embed_device))
        if attention_mask is not None:
            attention_mask = attention_mask.to(embed_device)

        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        if self.embed_dropout is not None:
            hidden_states = self.embed_dropout(hidden_states)

        if cache_position is None:
            cache_position = self._get_cache_position(hidden_states.shape[1], hidden_states.device)
        if position_ids is None:
            position_ids = cast(torch.LongTensor, cache_position.unsqueeze(0))

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        attention_mask_mapping = _build_attention_mask_mapping(
            self.config,
            hidden_states,
            attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Encoder transformer layers
        for layer in self.layers:
            layer_attention_mask = attention_mask_mapping.get(
                getattr(layer, "attention_type", "full_attention"),
                attention_mask_mapping["full_attention"],
            )
            layer_output = layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = _unwrap_layer_output(layer_output)
        
        return hidden_states, position_embeddings, position_ids, cache_position, past_key_values if use_cache else None

    def save_weights(self, path: str | Path):
        """Save encoder block weights to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"EncoderBlock weights saved to {path}")
    
    def load_weights(self, path: str | Path, map_location: str = "cpu"):
        """Load encoder block weights from file"""
        path = Path(path)
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"EncoderBlock weights loaded from {path}")

class SelectiveGate(nn.Module):
    """Input-dependent selective gate for reasoning recursion.

    Applies the ZOH-discretized SSM update:
        A_bar = exp(delta * A)           where A < 0 (always negative)
        h_new = A_bar * h_old + (1 - A_bar) * x_init

    - delta is input-dependent (computed from h_old and h_new), controlling how
      much of x_init to inject at each token/channel.
    - A is a learnable per-channel decay rate.
    - B projects x_init before injection.

    Initialized near identity (A_bar ≈ 1) so training starts from
    pretrained performance and gradually learns to use x_init.
    """

    def __init__(self, hidden_size: int, dt_rank: Optional[int] = None,
                 dt_min: float = 0.001, dt_max: float = 0.1, dt_scale: float = 1.0,
                 dt_init_floor: float = 1e-4):
        super().__init__()

        # Low-rank delta projection (as in Mamba)
        if dt_rank is None:
            dt_rank = math.ceil(hidden_size / 16)
        self.dt_rank = dt_rank
        self.delta_proj = nn.Linear(dt_rank, hidden_size, bias=True)

        # S4D real initialization
        A = torch.arange(1, hidden_size + 1, dtype=torch.float32)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # Input projection to dt_rank (reduces h_old → low-rank before delta_proj)
        self.dt_input_proj = nn.Linear(hidden_size, dt_rank, bias=False)

        # ── Mamba-style delta initialization ──
        # Weight: variance-preserving init
        dt_init_std = dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.delta_proj.weight, -dt_init_std, dt_init_std)

        # Bias: inverse-softplus of log-uniform samples in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(hidden_size) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus inverse
        with torch.no_grad():
            self.delta_proj.bias.copy_(inv_dt)

    def forward(self, h_new: torch.Tensor, h_old: torch.Tensor) -> torch.Tensor:
        orig_dtype = h_new.dtype
        param_dtype = self.dt_input_proj.weight.dtype
        h_new = h_new.to(param_dtype)
        h_old = h_old.to(param_dtype)
        dt_low = self.dt_input_proj(h_new-h_old)                # (B, T, dt_rank)
        delta = F.softplus(self.delta_proj(dt_low))       # (B, T, D), > 0
        A = -torch.exp(self.A_log)                         # always negative
        A_bar = torch.exp(delta * A)                       # ∈ (0, 1)
        out = A_bar * h_new + (1 - A_bar) * h_old
        return out.to(orig_dtype)

class ReasoningBlock(nn.Module):
    """Reasoning Block: configurable middle transformer layers.

    Args:
        config: The base model's ``PretrainedConfig``.
        layer_indices: List of layer indices included in this block.
        layers: Pre-extracted transformer layers (in order).
    """
    def __init__(self, config, layer_indices: list[int],
                 layers: list[nn.Module]):
        super().__init__()
        self.config = config
        self.layer_indices = sorted(layer_indices)
        self.layers = nn.ModuleList(layers)

        # Mamba-style selective state mixer for x_init re-injection
        hidden_size = config.hidden_size
        self.gate = SelectiveGate(hidden_size)
        self.gradient_checkpointing = False
 
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        q_head: Optional[nn.Module] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        hidden_old = hidden_states
        attention_mask_mapping = _build_attention_mask_mapping(
            self.config,
            hidden_states,
            attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        
        for layer in self.layers:
            layer_attention_mask = attention_mask_mapping.get(
                getattr(layer, "attention_type", "full_attention"),
                attention_mask_mapping["full_attention"],
            )
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._layer_forward,
                    layer, hidden_states, layer_attention_mask,
                    position_ids, position_embeddings, cache_position,
                    use_reentrant=False,
                )
            else:
                layer_output = layer(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
                hidden_states = _unwrap_layer_output(layer_output)

        q_hidden_states = hidden_states - hidden_old
        hidden_states = self.gate(hidden_states, hidden_old)

        if q_head is None:
            return (hidden_states, past_key_values) if use_cache else hidden_states

        # Pool the pre-gate delta so q evaluates the state change.
        if attention_mask is not None:
            token_mask = attention_mask
            if token_mask.dim() == 4:
                token_mask = token_mask[:, 0, 0, :]
            token_mask = token_mask.to(q_hidden_states.device)
            if token_mask.dtype != torch.bool:
                token_mask = token_mask != 0

            last_valid_idx = token_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(q_hidden_states.size(0), device=q_hidden_states.device)
            pooled = q_hidden_states[batch_idx, last_valid_idx, :]
        else:
            pooled = q_hidden_states[:, -1, :]

        q_params = list(q_head.parameters())
        if q_params:
            pooled = pooled.to(dtype=q_params[0].dtype)
        q_logit = q_head(pooled).squeeze(-1)
        if use_cache:
            return hidden_states, q_logit, past_key_values
        return hidden_states, q_logit

    @staticmethod
    def _layer_forward(layer, hidden_states, attention_mask, position_ids,
                       position_embeddings, cache_position):
        return _unwrap_layer_output(layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        ))

    def save_weights(self, path: str | Path):
        """Save plateau block weights to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"ReasoningBlock weights saved to {path}")
    
    def load_weights(self, path: str | Path, map_location: str = "cpu"):
        """Load plateau block weights from file"""
        path = Path(path)
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"ReasoningBlock weights loaded from {path}")

class DecoderBlock(nn.Module):
    """Decoder Block: configurable transformer layers + norm + lm_head.

    Args:
        config: The base model's ``PretrainedConfig``.
        layer_indices: List of layer indices included in this block.
        layers: Pre-extracted transformer layers (in order).
        norm: The final normalisation module from the base model.
        lm_head: The language-model head from the base model.
    """
    def __init__(self, config, layer_indices: list[int],
                 layers: list[nn.Module], norm: nn.Module,
                 lm_head: nn.Module):
        super().__init__()
        self.config = config
        self.layer_indices = sorted(layer_indices)
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        decoder_dtype = _first_parameter_dtype(self.layers, self.norm, self.lm_head)
        if decoder_dtype is not None and hidden_states.dtype != decoder_dtype:
            hidden_states = hidden_states.to(dtype=decoder_dtype)

        attention_mask_mapping = _build_attention_mask_mapping(
            self.config,
            hidden_states,
            attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Decoder transformer layers
        for layer in self.layers:
            layer_attention_mask = attention_mask_mapping.get(
                getattr(layer, "attention_type", "full_attention"),
                attention_mask_mapping["full_attention"],
            )
            layer_output = layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = _unwrap_layer_output(layer_output)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        return (logits, past_key_values) if use_cache else logits

    def save_weights(self, path: str | Path):
        """Save decoder block weights to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"DecoderBlock weights saved to {path}")
    
    def load_weights(self, path: str | Path, map_location: str = "cpu"):
        """Load decoder block weights from file"""
        path = Path(path)
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"DecoderBlock weights loaded from {path}")

class LDSForCausalLM(nn.Module, GenerationMixin):
    """Combined Model: Encoder + Reasoning + Decoder with Q-head.

    The reasoning block can be applied repeatedly via *run_recursion* to refine
    hidden states before decoding.

    Construction
    ------------
    * ``LDSForCausalLM(config)`` — builds the architecture with a
      randomly-initialised base model (useful when you will immediately
      load a checkpoint on top).
    * ``LDSForCausalLM.from_pretrained(config)`` — builds the architecture
      **and** loads pretrained base-model weights (the normal training
      starting point).
    """

    config_class = LDSConfig
    main_input_name = "input_ids"
    _supports_cache_class = True
    _is_stateful = False

    def __init__(
        self,
        config: LDSConfig,
        base_model=None,
        tokenizer=None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.N = config.N
        self.q_threshold = config.q_threshold
        self.q_eval_interval = max(1, int(getattr(config, "q_eval_interval", 1)))
        self.halting_strategy = getattr(config, "halting_strategy", "threshold")
        self.convergence_epsilon = getattr(config, "convergence_epsilon", 1e-2)
        self._causal_mask_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._runtime_stats_enabled = False
        self.reset_runtime_stats()

        # GenerationMixin needs a GenerationConfig
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
        )

        # Build from a supplied base model, or create one from config
        _created = False
        if base_model is None:
            base_config = config.get_base_config()
            base_model = AutoModelForCausalLM.from_config(base_config)
            _created = True

        self._init_from_base_model(config, base_model)

        if _created:
            del base_model

    # ------------------------------------------------------------------
    # Block construction
    # ------------------------------------------------------------------

    def _init_from_base_model(self, config: LDSConfig, base_model) -> None:
        """Extract layers from *base_model* and wire up the three blocks."""
        inner = base_model.model          # works for Llama / Qwen / Mistral / Gemma / Phi …
        base_cfg = base_model.config
        embed_tokens = _resolve_submodule(inner, ("embed_tokens",))
        rotary_emb = _resolve_submodule(inner, ("rotary_emb",))
        embed_dropout = _resolve_submodule(inner, ("embed_dropout",), required=False)
        final_norm = _resolve_submodule(inner, ("norm", "final_layernorm", "ln_f"))

        enc_indices = config.encoder_layer_indices
        rea_indices = config.reasoning_layer_indices
        dec_indices = config.decoder_layer_indices

        print(
            f"Decomposing model ({config.num_hidden_layers} layers): "
            f"encoder={enc_indices}, reasoning={rea_indices}, decoder={dec_indices}"
        )

        self.encoder = EncoderBlock(
            config=base_cfg,
            layer_indices=enc_indices,
            embed_tokens=embed_tokens,
            rotary_emb=rotary_emb,
            embed_dropout=embed_dropout,
            layers=[inner.layers[i] for i in enc_indices],
        )
        self.reasoning = ReasoningBlock(
            config=base_cfg,
            layer_indices=rea_indices,
            layers=[inner.layers[i] for i in rea_indices],
        )
        self.decoder = DecoderBlock(
            config=base_cfg,
            layer_indices=dec_indices,
            layers=[inner.layers[i] for i in dec_indices],
            norm=final_norm,
            lm_head=base_model.lm_head,
        )

        hidden_size = base_cfg.hidden_size
        self.q_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def _get_attn_implementation(self) -> Optional[str]:
        return getattr(self.encoder.config, "_attn_implementation", None)

    def _set_attn_implementation(self, value: Optional[str]) -> None:
        for block in (self.encoder, self.reasoning, self.decoder):
            if hasattr(block, "config") and block.config is not None:
                block.config._attn_implementation = value

    @classmethod
    def from_pretrained(
        cls,
        config_or_path,
        tokenizer=None,
        **model_kwargs,
    ) -> "LDSForCausalLM":
        """Load an LDS model.

        Supports two modes:

        1. **From base model** (original behaviour): pass an
           :class:`LDSConfig` instance.  The base model is downloaded
           from ``config.base_model_name_or_path`` and decomposed.
        2. **From saved directory / HF Hub repo**: pass a path string
           (local directory or Hub ``repo_id``) that contains a
           ``config.json`` + ``model.safetensors``.

        Args:
            config_or_path: Either an ``LDSConfig`` instance *or* a
                directory / Hub repo ID string.
            tokenizer: Optional tokenizer override.
            **model_kwargs: Forwarded to
                ``AutoModelForCausalLM.from_pretrained`` when building
                from a base model, or used for ``torch_dtype`` /
                ``device_map`` when loading from a saved directory.
        """
        # --- Mode 2: load from saved directory / HF Hub ---
        if isinstance(config_or_path, str):
            return cls._load_saved(config_or_path, tokenizer=tokenizer, **model_kwargs)

        # --- Mode 1: build from base model (original path) ---
        config: LDSConfig = config_or_path
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            config=config.get_base_config(),
            **model_kwargs,
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path,
            )
        model = cls(config, base_model=base_model, tokenizer=tokenizer)
        del base_model
        torch.cuda.empty_cache()
        return model

    @classmethod
    def _load_saved(
        cls,
        path_or_repo: str,
        tokenizer=None,
        **kwargs,
    ) -> "LDSForCausalLM":
        """Load from a ``save_pretrained`` directory or HF Hub repo."""
        from huggingface_hub import snapshot_download

        # Resolve to local directory
        local_dir = path_or_repo
        if not os.path.isdir(local_dir):
            local_dir = snapshot_download(repo_id=path_or_repo)

        config = LDSConfig.from_pretrained(local_dir)

        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        # Consume remaining kwargs to avoid unexpected keyword errors
        kwargs.pop("attn_implementation", None)

        # Build architecture (random weights) then load saved weights
        model = cls(config)

        weights_path = os.path.join(local_dir, "model.safetensors")
        if os.path.isfile(weights_path):
            state_dict = safetensors_load(weights_path)
        else:
            pt_path = os.path.join(local_dir, "combined_model.pt")
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

        cleaned = {k.replace(".module.", "."): v for k, v in state_dict.items()}
        cleaned = _remap_state_dict_keys(cleaned)
        model.load_state_dict(cleaned)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model.tokenizer = tokenizer

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        # Move model to the requested device
        if device_map is not None:
            if device_map == "auto":
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    from accelerate import dispatch_model, infer_auto_device_map

                    inferred_device_map = infer_auto_device_map(
                        model,
                        max_memory=max_memory,
                    )
                    model = dispatch_model(model, device_map=inferred_device_map)
                    print(f"[_load_saved] Model dispatched across devices: {inferred_device_map}")
                    return model

                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = device_map
            model = model.to(target_device)
            print(f"[_load_saved] Model moved to {target_device}")

        return model

    # ------------------------------------------------------------------
    # Recursion helpers
    # ------------------------------------------------------------------

    def _reasoning_kwargs(
        self,
        position_embeddings,
        position_ids,
        cache_position,
        attention_mask=None,
        hidden_states=None,
    ) -> dict:
        """Build the common keyword dict passed to every ``reasoning(...)`` call.
        """
        return dict(
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

    def _init_past_key_values(self) -> dict[str, Any]:
        return {
            "encoder": DynamicCache(),
            "reasoning": [DynamicCache() for _ in range(self.N)],
            "decoder": DynamicCache(),
            "seen_tokens": 0,
        }

    def _ensure_past_key_values(self, past_key_values: Optional[dict[str, Any]]) -> dict[str, Any]:
        if past_key_values is None:
            return self._init_past_key_values()

        normalized = dict(past_key_values)
        normalized.setdefault("encoder", DynamicCache())
        normalized.setdefault("reasoning", [DynamicCache() for _ in range(self.N)])
        normalized.setdefault("decoder", DynamicCache())
        normalized.setdefault("seen_tokens", 0)

        if len(normalized["reasoning"]) < self.N:
            normalized["reasoning"] = list(normalized["reasoning"]) + [DynamicCache() for _ in range(self.N - len(normalized["reasoning"]))]
        elif len(normalized["reasoning"]) > self.N:
            normalized["reasoning"] = list(normalized["reasoning"][:self.N])

        return normalized

    def _crop_past_key_values(self, past_key_values: Optional[dict[str, Any]], max_length: int):
        if past_key_values is None or max_length < 0:
            return past_key_values

        cropped = self._ensure_past_key_values(past_key_values)
        cropped["encoder"] = _crop_cache_object(cropped["encoder"], max_length)
        cropped["reasoning"] = [_crop_cache_object(reasoning_cache, max_length) for reasoning_cache in cropped["reasoning"]]
        cropped["decoder"] = _crop_cache_object(cropped["decoder"], max_length)
        cropped["seen_tokens"] = max_length
        return cropped

    def set_runtime_stats_enabled(self, enabled: bool) -> None:
        self._runtime_stats_enabled = bool(enabled)

    def reset_runtime_stats(self) -> None:
        self._runtime_stats = {
            "forward_calls": 0,
            "total_reasoning_steps": 0,
            "total_q_evaluations": 0,
            "early_stop_count": 0,
            "reasoning_steps_histogram": Counter(),
            "q_evaluations_histogram": Counter(),
            "final_q_min_sum": 0.0,
            "final_q_min_count": 0,
        }

    def _update_runtime_stats(
        self,
        *,
        reasoning_steps: int,
        q_evaluations: int,
        early_stopped: bool,
        final_q_min: float | None,
    ) -> None:
        if not self._runtime_stats_enabled:
            return

        stats = self._runtime_stats
        stats["forward_calls"] += 1
        stats["total_reasoning_steps"] += int(reasoning_steps)
        stats["total_q_evaluations"] += int(q_evaluations)
        stats["reasoning_steps_histogram"][int(reasoning_steps)] += 1
        stats["q_evaluations_histogram"][int(q_evaluations)] += 1
        if early_stopped:
            stats["early_stop_count"] += 1
        if final_q_min is not None:
            stats["final_q_min_sum"] += float(final_q_min)
            stats["final_q_min_count"] += 1

    def get_runtime_stats(self) -> dict[str, object]:
        stats = self._runtime_stats
        forward_calls = int(stats["forward_calls"])
        final_q_min_count = int(stats["final_q_min_count"])

        def _sorted_histogram(counter: Counter) -> dict[str, int]:
            return {str(k): int(counter[k]) for k in sorted(counter)}

        return {
            "enabled": self._runtime_stats_enabled,
            "forward_calls": forward_calls,
            "avg_reasoning_steps": (
                float(stats["total_reasoning_steps"]) / forward_calls if forward_calls else 0.0
            ),
            "avg_q_evaluations": (
                float(stats["total_q_evaluations"]) / forward_calls if forward_calls else 0.0
            ),
            "early_stop_count": int(stats["early_stop_count"]),
            "early_stop_rate": (
                float(stats["early_stop_count"]) / forward_calls if forward_calls else 0.0
            ),
            "reasoning_steps_histogram": _sorted_histogram(stats["reasoning_steps_histogram"]),
            "q_evaluations_histogram": _sorted_histogram(stats["q_evaluations_histogram"]),
            "mean_final_q_min": (
                float(stats["final_q_min_sum"]) / final_q_min_count if final_q_min_count else None
            ),
        }

    def run_outer_no_q(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run *t_steps* outer iterations, each with *n_latent* plateau calls.

        No Q-head evaluation is performed — this is the "cheap" recursion used
        for the first ``T − 1`` outer steps.
        """
        hidden_states = self.reasoning(hidden_states=hidden_states, **kwargs)
        
        return hidden_states

    def run_final_with_q(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the last outer step: ``n_latent − 1`` plateau calls followed by
        one call that also evaluates the Q-head.

        Returns ``(hidden_states, q_logit)``.
        """
        hidden_states = self.reasoning(hidden_states=hidden_states, **kwargs)
        
        hidden_states, q_logit = self.reasoning(
            hidden_states=hidden_states,
            q_head=self.q_head,
            **kwargs,
        )
        
        return hidden_states, q_logit

    def run_recursion(
        self,
        hidden_states: torch.Tensor,

        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one full supervision step: ``T`` outer iterations of recursion.

        Equivalent to ``run_outer_no_q(T-1)`` followed by
        ``run_final_with_q()``.

        Returns ``(hidden_states, q_logit)``.
        """
        hidden_states = self.run_outer_no_q(hidden_states, **kwargs)
        return self.run_final_with_q(hidden_states, **kwargs)

    # ------------------------------------------------------------------
    # Properties required by GenerationMixin
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def can_generate() -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    # ------------------------------------------------------------------
    # Forward / generation
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[dict[str, Any]] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> LDSCausalLMOutput:
        past_key_values = self._ensure_past_key_values(past_key_values) if use_cache else None
        past_seen_tokens = int(past_key_values["seen_tokens"]) if past_key_values is not None else 0

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
        if position_ids is None:
            position_ids = cast(torch.LongTensor, cache_position.unsqueeze(0))

        hidden_states, position_embeddings, position_ids, cache_position, next_encoder_past = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values["encoder"] if past_key_values is not None else None,
            use_cache=use_cache,
        )

        original_attn_implementation = self._get_attn_implementation()
        use_eager_fallback = (
            use_cache
            and original_attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.shape[0] > 1
            and bool((attention_mask == 0).any().item())
        )

        if use_eager_fallback:
            self._set_attn_implementation("eager")

        try:
            reasoning_kwargs = self._reasoning_kwargs(
                position_embeddings, position_ids, cache_position, attention_mask,
                hidden_states=hidden_states,
            )

            q_head_dtype = _first_parameter_dtype(self.q_head)
            q_eval_interval = max(1, int(self.q_eval_interval))
            executed_reasoning_steps = 0
            q_evaluations = 0
            early_stopped = False
            final_q_min: float | None = None
            next_reasoning_past: list[list[Any]] = []
            halting_strategy = getattr(self, "halting_strategy", "threshold")
            prev_hidden_last: torch.Tensor | None = None
            # Sample-wise halting is only safe on the no-cache path.
            # With generation KV caches enabled, each recursion depth cache keeps a
            # dense batch/time layout, so some samples cannot stop advancing without
            # introducing per-sample cache length skew.
            samplewise_halting = not use_cache
            active_mask = torch.ones(hidden_states.size(0), dtype=torch.bool, device=hidden_states.device)
            cached_first_halt_mask = torch.zeros(hidden_states.size(0), dtype=torch.bool, device=hidden_states.device)
            cached_first_halt_hidden_states = hidden_states
            # CDF: accumulate log(1 - lambda_j) for survival probability
            log_survival = torch.zeros(hidden_states.size(0), device=hidden_states.device, dtype=hidden_states.dtype)

            for recursion_idx in range(self.N):
                if samplewise_halting and not bool(active_mask.any().item()):
                    early_stopped = recursion_idx < (self.N - 1)
                    break

                step_active_mask = active_mask
                hidden_states_before = hidden_states

                if halting_strategy == "convergence":
                    prev_hidden_last = hidden_states[:, -1, :].detach().clone()

                reasoning_output = self.reasoning(
                    hidden_states=hidden_states,
                    past_key_values=past_key_values["reasoning"][recursion_idx] if past_key_values is not None else None,
                    use_cache=use_cache,
                    **reasoning_kwargs,
                )
                if use_cache:
                    next_hidden_states, current_reasoning_past = reasoning_output
                    next_reasoning_past.append(current_reasoning_past)
                else:
                    next_hidden_states = reasoning_output

                if samplewise_halting:
                    hidden_states = torch.where(
                        step_active_mask.view(-1, 1, 1),
                        next_hidden_states,
                        hidden_states_before,
                    )
                else:
                    hidden_states = next_hidden_states

                executed_reasoning_steps = recursion_idx + 1

                # --- Strategy: convergence (no q_head needed) ---
                if halting_strategy == "convergence":
                    cur_hidden_last = hidden_states[:, -1, :].detach()
                    delta = (cur_hidden_last - prev_hidden_last).norm(dim=-1)
                    epsilon = getattr(self, "convergence_epsilon", 1e-2)
                    if samplewise_halting:
                        newly_halted = step_active_mask & (delta < epsilon)
                        active_mask = step_active_mask & ~newly_halted
                        if not bool(active_mask.any().item()):
                            early_stopped = recursion_idx < (self.N - 1)
                            break
                    else:
                        newly_halted = (~cached_first_halt_mask) & (delta < epsilon)
                        if bool(newly_halted.any().item()):
                            cached_first_halt_hidden_states = torch.where(
                                newly_halted.view(-1, 1, 1),
                                hidden_states,
                                cached_first_halt_hidden_states,
                            )
                            cached_first_halt_mask = cached_first_halt_mask | newly_halted
                        if float(delta.max().item()) < epsilon:
                            early_stopped = recursion_idx < (self.N - 1)
                            break
                    continue

                should_evaluate_q = ((recursion_idx + 1) % q_eval_interval == 0) or (recursion_idx == self.N - 1)
                if not should_evaluate_q:
                    continue

                q_evaluations += 1
                q_input = hidden_states[:, -1, :]
                q_head_device = None
                try:
                    q_head_device = next(self.q_head.parameters()).device
                except StopIteration:
                    q_head_device = None
                if q_head_device is not None and q_input.device != q_head_device:
                    q_input = q_input.to(q_head_device)
                if q_head_dtype is not None and q_input.dtype != q_head_dtype:
                    q_input = q_input.to(dtype=q_head_dtype)
                q_val = torch.sigmoid(self.q_head(q_input).squeeze(-1))
                if q_val.device != hidden_states.device:
                    q_val = q_val.to(hidden_states.device)

                if halting_strategy == "cdf":
                    if samplewise_halting:
                        active_q = q_val[step_active_mask]
                        if active_q.numel() > 0:
                            log_survival[step_active_mask] = (
                                log_survival[step_active_mask]
                                + torch.log1p(-active_q.clamp(max=1.0 - 1e-7))
                            )
                            cdf_val = 1.0 - torch.exp(log_survival)
                            active_cdf = cdf_val[step_active_mask]
                            final_q_min = float(active_cdf.min().item())
                            newly_halted = step_active_mask & (cdf_val > self.q_threshold)
                            active_mask = step_active_mask & ~newly_halted
                            if not bool(active_mask.any().item()):
                                early_stopped = recursion_idx < (self.N - 1)
                                break
                    else:
                        # lambda_j = q_val (per-sample hazard rate)
                        # Update log-survival: log(1 - lambda_j), clamped for stability
                        log_survival = log_survival + torch.log1p(-q_val.clamp(max=1.0 - 1e-7))
                        # CDF = 1 - exp(log_survival)
                        cdf_val = 1.0 - torch.exp(log_survival)
                        cdf_min = float(cdf_val.min().item())
                        final_q_min = cdf_min
                        newly_halted = (~cached_first_halt_mask) & (cdf_val > self.q_threshold)
                        if bool(newly_halted.any().item()):
                            cached_first_halt_hidden_states = torch.where(
                                newly_halted.view(-1, 1, 1),
                                hidden_states,
                                cached_first_halt_hidden_states,
                            )
                            cached_first_halt_mask = cached_first_halt_mask | newly_halted
                        if cdf_min > self.q_threshold:
                            early_stopped = recursion_idx < (self.N - 1)
                            break
                else:
                    if samplewise_halting:
                        active_q = q_val[step_active_mask]
                        if active_q.numel() > 0:
                            final_q_min = float(active_q.min().item())
                            newly_halted = step_active_mask & (q_val > self.q_threshold)
                            active_mask = step_active_mask & ~newly_halted
                            if not bool(active_mask.any().item()):
                                early_stopped = recursion_idx < (self.N - 1)
                                break
                    else:
                        # Default: threshold strategy
                        final_q_min = float(q_val.min().item())
                        newly_halted = (~cached_first_halt_mask) & (q_val > self.q_threshold)
                        if bool(newly_halted.any().item()):
                            cached_first_halt_hidden_states = torch.where(
                                newly_halted.view(-1, 1, 1),
                                hidden_states,
                                cached_first_halt_hidden_states,
                            )
                            cached_first_halt_mask = cached_first_halt_mask | newly_halted
                        if final_q_min > self.q_threshold:
                            early_stopped = recursion_idx < (self.N - 1)
                            break

            if use_cache and bool(cached_first_halt_mask.any().item()):
                hidden_states = torch.where(
                    cached_first_halt_mask.view(-1, 1, 1),
                    cached_first_halt_hidden_states,
                    hidden_states,
                )

            self._update_runtime_stats(
                reasoning_steps=executed_reasoning_steps,
                q_evaluations=q_evaluations,
                early_stopped=early_stopped,
                final_q_min=final_q_min,
            )

            decoder_output = self.decoder(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values["decoder"] if past_key_values is not None else None,
                use_cache=use_cache,
            )
        finally:
            if use_eager_fallback:
                self._set_attn_implementation(original_attn_implementation)
        if use_cache:
            logits, next_decoder_past = decoder_output
            next_past_key_values = {
                "encoder": next_encoder_past,
                "reasoning": next_reasoning_past,
                "decoder": next_decoder_past,
                "seen_tokens": past_seen_tokens + input_ids.shape[1],
            }
        else:
            logits = decoder_output
            next_past_key_values = None

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return LDSCausalLMOutput(loss=loss, logits=logits, past_key_values=next_past_key_values)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        global_step: int,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        max_checkpoints: int = 0,
        state_dict_override: Optional[dict] = None,
    ) -> str:
        """Save a training checkpoint and return the save directory path.

        Maintains a ``latest`` symlink and prunes old checkpoints when
        *max_checkpoints* > 0.

        When *state_dict_override* is provided (e.g. a pre-gathered full
        state dict from FSDP / DeepSpeed), it is saved directly instead
        of calling ``self.state_dict()``.
        """
        save_dir = os.path.join(checkpoint_dir, f"step_{global_step}")
        os.makedirs(save_dir, exist_ok=True)

        sd = state_dict_override if state_dict_override is not None else self.state_dict()
        torch.save(sd, os.path.join(save_dir, "combined_model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        torch.save(
            {"global_step": global_step, "epoch": epoch},
            os.path.join(save_dir, "training_state.pt"),
        )

        self._update_latest_symlink(checkpoint_dir, global_step)
        print(f"[checkpoint] Saved checkpoint at step {global_step} -> {save_dir}")

        if max_checkpoints > 0:
            self._prune_old_checkpoints(checkpoint_dir, max_checkpoints)

        return save_dir

    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str = "cpu",
    ) -> dict:
        """Load a training checkpoint from disk.

        Returns the stored training-state dict (``global_step``, ``epoch``).
        """
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        model_path = os.path.join(checkpoint_path, "combined_model.pt")
        state_dict = torch.load(model_path, map_location=map_location, weights_only=True)
        # Strip DDP `.module.` prefix saved by accelerate/DDP training
        cleaned = {k.replace(".module.", "."): v for k, v in state_dict.items()}
        # Backward-compat: remap old key prefixes → current names
        cleaned = _remap_state_dict_keys(cleaned)
        self.load_state_dict(cleaned)

        if optimizer is not None:
            opt_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=map_location, weights_only=True)
                )

        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(
                state_path, map_location=map_location, weights_only=True,
            )
        else:
            training_state = {"global_step": 0, "epoch": 0}

        print(
            f"[checkpoint] Loaded checkpoint from {checkpoint_path} "
            f"(step={training_state['global_step']}, epoch={training_state['epoch']})"
        )
        return training_state

    @staticmethod
    def resolve_resume_path(resume: str, checkpoint_dir: str) -> str:
        """Resolve a *resume* value (e.g. ``"latest"``) to a real directory path."""
        resolved = resume
        if resume == "latest":
            resolved = os.path.join(checkpoint_dir, "latest")
        if os.path.islink(resolved):
            resolved = os.path.join(os.path.dirname(resolved), os.readlink(resolved))
        return resolved

    @staticmethod
    def _update_latest_symlink(checkpoint_dir: str, global_step: int) -> None:
        latest_link = os.path.join(checkpoint_dir, "latest")
        if os.path.islink(latest_link):
            os.remove(latest_link)
        elif os.path.exists(latest_link):
            shutil.rmtree(latest_link)
        os.symlink(f"step_{global_step}", latest_link)

    @staticmethod
    def _prune_old_checkpoints(checkpoint_dir: str, max_checkpoints: int) -> None:
        existing: list[tuple[int, str]] = []
        for name in os.listdir(checkpoint_dir):
            if not name.startswith("step_"):
                continue
            full = os.path.join(checkpoint_dir, name)
            if not os.path.isdir(full) or os.path.islink(full):
                continue
            try:
                step_num = int(name.split("_", 1)[1])
                existing.append((step_num, full))
            except ValueError:
                continue
        existing.sort(key=lambda x: x[0])
        while len(existing) > max_checkpoints:
            _, old_dir = existing.pop(0)
            shutil.rmtree(old_dir)
            print(f"[checkpoint] Deleted old checkpoint: {old_dir}")

    # ------------------------------------------------------------------
    # Generation (overrides GenerationMixin.generate)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        generation_config: GenerationConfig | None = None,
        max_context: int = 1024,
        **kwargs,
    ) -> torch.LongTensor:
        """Generate token-by-token with q-head early stopping in reasoning.

        Accepts standard HuggingFace generation arguments.  The reasoning
        loop's q-head early exit is applied at every forward pass.

        Returns:
            ``(B, L + generated)`` token ids.
        """
        # --- Resolve generation parameters ---
        gc = generation_config or self.generation_config or GenerationConfig()
        explicit_max_new_tokens = max_new_tokens
        resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else getattr(gc, "max_new_tokens", None)
        if resolved_max_new_tokens is None:
            resolved_max_new_tokens = 256

        resolved_do_sample = do_sample if do_sample is not None else getattr(gc, "do_sample", True)
        resolved_temperature = temperature if temperature is not None else getattr(gc, "temperature", None)
        resolved_top_k = top_k if top_k is not None else getattr(gc, "top_k", None)
        resolved_top_p = top_p if top_p is not None else getattr(gc, "top_p", None)
        resolved_repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else getattr(gc, "repetition_penalty", None)
        )

        max_new_tokens = int(resolved_max_new_tokens)
        do_sample = bool(resolved_do_sample)
        temperature = 1.0 if resolved_temperature is None else float(resolved_temperature)
        top_k = 50 if resolved_top_k is None else int(resolved_top_k)
        top_p = 1.0 if resolved_top_p is None else float(resolved_top_p)
        repetition_penalty = 1.0 if resolved_repetition_penalty is None else float(resolved_repetition_penalty)

        if eos_token_id is None:
            eos_token_id = getattr(gc, "eos_token_id", None)
            if eos_token_id is None and self.tokenizer is not None:
                eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = getattr(gc, "pad_token_id", None)
            if pad_token_id is None and self.tokenizer is not None:
                pad_token_id = self.tokenizer.pad_token_id

        if max_length is not None and explicit_max_new_tokens is None:
            max_new_tokens = max_length - input_ids.shape[1]

        generated = cast(torch.LongTensor, input_ids.clone())
        batch_size_gen = input_ids.shape[0]
        finished = torch.zeros(batch_size_gen, dtype=torch.bool, device=input_ids.device)
        past_key_values = None

        for _ in range(max_new_tokens):
            active_context_len = min(generated.shape[1], max_context)
            if past_key_values is not None and active_context_len == max_context:
                past_key_values = self._crop_past_key_values(past_key_values, max_context - 1)

            if past_key_values is None:
                model_input = generated[:, -max_context:] if generated.shape[1] > max_context else generated
                if attention_mask is not None:
                    mask = attention_mask[:, -model_input.shape[1]:].contiguous()
                else:
                    mask = torch.ones_like(model_input)
            else:
                model_input = generated[:, -1:]
                if attention_mask is not None:
                    mask = attention_mask[:, -active_context_len:].contiguous()
                else:
                    mask = torch.ones((batch_size_gen, active_context_len), dtype=torch.long, device=generated.device)

            output = self(
                input_ids=cast(torch.LongTensor, model_input),
                attention_mask=mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_logits = output.logits[:, -1, :].float()

            # --- Repetition penalty ---
            if repetition_penalty != 1.0:
                for b in range(batch_size_gen):
                    if finished[b]:
                        continue
                    prev_tokens = generated[b].unique()
                    penalty_logits = next_logits[b, prev_tokens]
                    next_logits[b, prev_tokens] = torch.where(
                        penalty_logits > 0,
                        penalty_logits / repetition_penalty,
                        penalty_logits * repetition_penalty,
                    )

            # --- Temperature ---
            if do_sample and temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature

            if do_sample:
                # --- Top-k filtering ---
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

                # --- Top-p (nucleus) filtering ---
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[remove_mask] = float("-inf")
                    next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Replace tokens for already-finished samples with pad
            if pad_token_id is not None:
                finished_dev = finished.to(next_token.device)
                next_token = torch.where(
                    finished_dev.unsqueeze(1),
                    torch.tensor(pad_token_id, device=next_token.device, dtype=next_token.dtype),
                    next_token,
                )

            generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)

            # Update per-sample finished status
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1).to(finished.device) == eos_token_id)

            # Extend attention_mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size_gen, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                    dim=-1,
                )

            # --- EOS stopping: all samples finished ---
            if finished.all():
                break

        return cast(torch.LongTensor, generated)

    @torch.no_grad()
    def generate_simple(self, input_ids, max_new_tokens=100, temperature=0.7, top_k: int = 50, max_context: int = 1024):
        """Legacy simple generation helper. Prefer :meth:`generate`."""
        return self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            max_context=max_context,
        )

    # ------------------------------------------------------------------
    # HuggingFace-compatible save / push
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights, config, and tokenizer in HF-compatible format.

        Creates:
          - ``config.json``       (LDSConfig)
          - ``model.safetensors`` (all weights)
          - tokenizer files       (tokenizer_config.json, etc.)
        """
        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        # Config
        self.config.save_pretrained(save_directory)

        # Weights — clean DDP prefixes before saving
        state_dict = self.state_dict()
        cleaned = {k.replace(".module.", "."): v for k, v in state_dict.items()}
        safetensors_save(cleaned, os.path.join(save_directory, "model.safetensors"))

        # Tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        print(f"[save_pretrained] Saved to {save_directory}")

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload LDS model",
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Save and upload the model to the Hugging Face Hub.

        Returns the URL of the created/updated repo.
        """
        import tempfile
        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                commit_message=commit_message,
            )

        url = f"https://huggingface.co/{repo_id}"
        print(f"[push_to_hub] Uploaded to {url}")
        return url


__all__ = ["LDSForCausalLM", "_remap_state_dict_keys"]


