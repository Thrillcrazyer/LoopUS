"""LDS (Latent Deep-Supervision) model configuration."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


def _normalize_base_config_dict(base_config_dict: dict) -> dict:
    """Extract the text-model config from composite multimodal configs."""
    if "num_hidden_layers" in base_config_dict:
        return dict(base_config_dict)

    text_config = base_config_dict.get("text_config")
    if isinstance(text_config, dict) and "num_hidden_layers" in text_config:
        normalized = dict(text_config)
        normalized.setdefault("_name_or_path", base_config_dict.get("_name_or_path", ""))
        normalized.setdefault(
            "transformers_version",
            base_config_dict.get("transformers_version"),
        )
        return normalized

    raise KeyError("num_hidden_layers")


class LDSConfig(PretrainedConfig):
    """Configuration for :class:`LDSForCausalLM`.

    Stores the base-model identity, layer assignments for the three blocks
    (encoder / reasoning / decoder), and recursion hyper-parameters.

    The base model's own ``PretrainedConfig`` is stored as a plain dict
    (``base_config_dict``) so that the whole config round-trips through
    JSON without custom serialisation logic.  Use :meth:`get_base_config`
    to reconstruct the typed config object when needed.
    """

    model_type = "lds"

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen3-0.6B",
        base_config_dict: dict | None = None,
        encoder_layer_indices: list[int] | None = None,
        decoder_layer_indices: list[int] | None = None,
        reasoning_layer_indices: list[int] | None = None,
        N: int = 6,
        q_threshold: float = 0.9,
        q_eval_interval: int = 1,
        halting_strategy: str = "threshold",
        convergence_epsilon: float = 1e-2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path

        # Resolve base config ------------------------------------------------
        if base_config_dict is not None:
            self.base_config_dict = _normalize_base_config_dict(base_config_dict)
        else:
            self.base_config_dict = _normalize_base_config_dict(
                AutoConfig.from_pretrained(base_model_name_or_path).to_dict()
            )

        # Avoid "tied weights" warning when both embed_tokens and lm_head
        # are present in the checkpoint.
        self.base_config_dict["tie_word_embeddings"] = False

        num_layers: int = self.base_config_dict["num_hidden_layers"]
        self.hidden_size: int = self.base_config_dict["hidden_size"]
        self.vocab_size: int = self.base_config_dict.get("vocab_size", 0)

        # Layer assignments ---------------------------------------------------
        if encoder_layer_indices is None:
            encoder_layer_indices = []
        if decoder_layer_indices is None:
            decoder_layer_indices = []

        # Expand open-ended sentinel: [-N] → [N .. num_layers-1]
        if len(decoder_layer_indices) == 1 and decoder_layer_indices[0] < 0:
            start = -decoder_layer_indices[0]
            decoder_layer_indices = list(range(start, num_layers))
        if len(encoder_layer_indices) == 1 and encoder_layer_indices[0] < 0:
            start = -encoder_layer_indices[0]
            encoder_layer_indices = list(range(start, num_layers))

        enc_set = set(encoder_layer_indices)
        dec_set = set(decoder_layer_indices)
        overlap = enc_set & dec_set
        if overlap:
            raise ValueError(
                f"Encoder and decoder layers overlap: {sorted(overlap)}"
            )

        if reasoning_layer_indices is None:
            reasoning_layer_indices = sorted(
                set(range(num_layers)) - enc_set - dec_set
            )

        if not reasoning_layer_indices:
            raise ValueError(
                f"No layers left for reasoning block. "
                f"Total layers: {num_layers}, "
                f"encoder: {encoder_layer_indices}, "
                f"decoder: {decoder_layer_indices}"
            )

        self.encoder_layer_indices: list[int] = sorted(encoder_layer_indices)
        self.decoder_layer_indices: list[int] = sorted(decoder_layer_indices)
        self.reasoning_layer_indices: list[int] = sorted(reasoning_layer_indices)

        # Recursion hyper-parameters ------------------------------------------
        self.N: int = N
        self.q_threshold: float = q_threshold
        self.q_eval_interval: int = max(1, int(q_eval_interval))
        self.halting_strategy: str = halting_strategy
        self.convergence_epsilon: float = convergence_epsilon

    @property
    def num_hidden_layers(self) -> int:
        return self.base_config_dict["num_hidden_layers"]

    # -----------------------------------------------------------------
    # Base-config reconstruction
    # -----------------------------------------------------------------

    def get_base_config(self) -> PretrainedConfig:
        """Reconstruct the base model's typed ``PretrainedConfig``."""
        from transformers import CONFIG_MAPPING

        model_type = self.base_config_dict.get("model_type")
        if model_type and model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
        else:
            config_class = PretrainedConfig
        filtered = {
            k: v
            for k, v in self.base_config_dict.items()
            if k not in ("transformers_version",)
        }
        return config_class(**filtered)

__all__=["LDSConfig"]