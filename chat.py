"""Simple inference script for an SFT-trained LDS model.

Supports three loading modes:
1. HF-style LDS directory or Hub repo id.
2. Legacy checkpoint directory containing ``combined_model.pt`` plus an
   initializer LDS model (auto-inferred from recent SFT logs when possible).
3. Legacy checkpoint directory rebuilt from a base model + layer split.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

if os.name == "posix":
    import termios
    import tty

    _TTY_CONTROL_ERRORS: tuple[type[BaseException], ...] = (OSError, termios.error, ValueError)
else:
    termios = None
    tty = None
    _TTY_CONTROL_ERRORS = (OSError, ValueError)

import torch
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import AutoTokenizer

from models.configuration_lds import LDSConfig
from models.modeling_lds import LDSForCausalLM, _remap_state_dict_keys
from utils.common import parse_layer_indices


DEFAULT_CHECKPOINT_DIR = "Thrillcrazyer/SFT_LDS_QWEN1.7B"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_RUNTIME_STATS_PATH = "logs/chat_runtime_stats.jsonl"


@dataclass
class TokenLoopChip:
    text: str
    loop_count: int


@dataclass
class LiveGenerationState:
    response: str = ""
    token_count: int = 0
    current_loop_count: int = 0
    max_loop_count: int = 0
    current_chunk: str = ""
    phase: str = "Ready"
    token_chips: list[TokenLoopChip] = field(default_factory=list)
    aborted: bool = False


class EscapeKeyWatcher:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._abort_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._original_termios: list[Any] | None = None

    @property
    def supported(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        if not supports_dynamic_ui() or os.name != "posix":
            return
        try:
            fd = sys.stdin.fileno()
            original_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except _TTY_CONTROL_ERRORS:
            return

        self._fd = fd
        self._original_termios = original_termios
        self._thread = threading.Thread(target=self._listen, name="chat-esc-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.2)
        if self._fd is not None and self._original_termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_termios)
            except _TTY_CONTROL_ERRORS:
                pass
        self._thread = None
        self._fd = None
        self._original_termios = None

    def aborted(self) -> bool:
        return self._abort_event.is_set()

    def _listen(self) -> None:
        if self._fd is None:
            return
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                pressed = os.read(self._fd, 1)
            except OSError:
                return
            if pressed == b"\x1b":
                self._abort_event.set()
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple chat/inference for an SFT-trained LDS model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "LDS model path. Supports a save_pretrained directory, a HF Hub repo id, "
            "a legacy checkpoint directory, or a direct combined_model.pt path"
        ),
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help=(
            "Initializer LDS model used before SFT. Needed for legacy combined_model.pt "
            "unless it can be inferred from recent logs"
        ),
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=DEFAULT_LOGS_DIR,
        help="Directory scanned to infer the initializer LDS model from recent SFT logs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Base model name for rebuilding a legacy checkpoint when --init-model is unavailable",
    )
    parser.add_argument(
        "--encoder-layers",
        type=str,
        default=None,
        help="Encoder layer spec for legacy rebuild, e.g. '0,1' or 'none'",
    )
    parser.add_argument(
        "--decoder-layers",
        type=str,
        default=None,
        help="Decoder layer spec for legacy rebuild, e.g. '27' or '13-'",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run once")
    parser.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive REPL")
    parser.add_argument(
        "--runtime-stats-path",
        type=str,
        default=DEFAULT_RUNTIME_STATS_PATH,
        help="JSONL file where per-response runtime stats are appended",
    )
    parser.add_argument(
        "--no-runtime-stats",
        action="store_true",
        help="Disable recursion runtime stats printing and JSONL logging",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=8,
        help="How many recent user/assistant turns to keep in CLI chat mode",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Use the raw prompt string instead of tokenizer.apply_chat_template",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument("--n-recursion", type=int, default=8, help="Override LDS recursion depth")
    parser.add_argument(
        "--q-stop-threshold",
        type=float,
        default=0.6,
        help="Override LDS halting threshold",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def infer_init_model_from_logs(logs_dir: str) -> str | None:
    log_root = Path(logs_dir)
    if not log_root.is_dir():
        return None

    pattern = re.compile(r"\[prepare_model\] Loading LDS model from: (?P<model>.+)")
    candidates = sorted(log_root.glob("sft_train_*.log"), reverse=True)

    for log_path in candidates:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = pattern.search(text)
        if match:
            return match.group("model").strip()
    return None


def is_saved_lds_directory(path_str: str) -> bool:
    path = Path(path_str)
    return path.is_dir() and (path / "config.json").is_file()


def resolve_legacy_checkpoint_path(model_path: str) -> Path | None:
    path = Path(model_path)
    if path.is_file():
        return path
    candidate = path / "combined_model.pt"
    if candidate.is_file():
        return candidate
    return None


def load_state_dict_into_model(model: LDSForCausalLM, checkpoint_path: Path) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cleaned = {key.replace(".module.", "."): value for key, value in state_dict.items()}
    cleaned = _remap_state_dict_keys(cleaned)
    incompatible = model.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Failed to load checkpoint cleanly. "
            f"missing_keys={incompatible.missing_keys[:10]}, "
            f"unexpected_keys={incompatible.unexpected_keys[:10]}"
        )


def load_from_base_model(args: argparse.Namespace, torch_dtype: torch.dtype, device_map: str) -> LDSForCausalLM:
    if not args.model_name:
        raise ValueError(
            "Legacy combined_model.pt without --init-model needs --model-name "
            "and the original layer split"
        )

    config = LDSConfig(
        base_model_name_or_path=args.model_name,
        encoder_layer_indices=parse_layer_indices(args.encoder_layers),
        decoder_layer_indices=parse_layer_indices(args.decoder_layers),
        N=args.n_recursion or 8,
    )
    return LDSForCausalLM.from_pretrained(
        config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )


def load_model(args: argparse.Namespace) -> tuple[LDSForCausalLM, AutoTokenizer]:
    torch_dtype = resolve_dtype(args.dtype)
    device_map = resolve_device(args.device)
    model_path = args.model_path
    checkpoint_path = resolve_legacy_checkpoint_path(model_path)

    if is_saved_lds_directory(model_path):
        print(f"[load] Loading saved LDS directory: {model_path}")
        model = LDSForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        tokenizer = model.tokenizer
    elif checkpoint_path is not None:
        init_model = args.init_model or infer_init_model_from_logs(args.logs_dir)
        if init_model:
            print(f"[load] Legacy checkpoint: {checkpoint_path}")
            print(f"[load] Initializer LDS model: {init_model}")
            model = LDSForCausalLM.from_pretrained(
                init_model,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            print(f"[load] Legacy checkpoint: {checkpoint_path}")
            print("[load] No initializer model inferred from logs. Falling back to base model rebuild.")
            model = load_from_base_model(args, torch_dtype=torch_dtype, device_map=device_map)
        load_state_dict_into_model(model, checkpoint_path)
        tokenizer = model.tokenizer
    else:
        print(f"[load] Loading LDS model from Hub: {model_path}")
        model = LDSForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        tokenizer = model.tokenizer

    if tokenizer is None:
        tokenizer_source = args.init_model or args.model_name or model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        model.tokenizer = tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.n_recursion is not None:
        model.N = args.n_recursion
        model.config.N = args.n_recursion
    if args.q_stop_threshold is not None:
        model.q_threshold = args.q_stop_threshold
        model.config.q_threshold = args.q_stop_threshold

    model.set_runtime_stats_enabled(not args.no_runtime_stats)
    model.reset_runtime_stats()
    model.eval()
    return model, tokenizer


def get_input_device(model: LDSForCausalLM) -> torch.device:
    return cast(torch.device, model.encoder.embed_tokens.weight.device)


def get_runtime_stats_snapshot(model: LDSForCausalLM) -> dict[str, Any]:
    return cast(dict[str, Any], model.get_runtime_stats())


def diff_runtime_stats(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_calls = int(before.get("forward_calls", 0))
    after_calls = int(after.get("forward_calls", 0))
    delta_calls = max(0, after_calls - before_calls)

    before_hist = cast(dict[str, int], before.get("reasoning_steps_histogram", {}))
    after_hist = cast(dict[str, int], after.get("reasoning_steps_histogram", {}))
    delta_hist: dict[str, int] = {}
    delta_steps = 0
    for key, after_count in after_hist.items():
        diff = int(after_count) - int(before_hist.get(key, 0))
        if diff > 0:
            delta_hist[key] = diff
            delta_steps += int(key) * diff

    before_q_hist = cast(dict[str, int], before.get("q_evaluations_histogram", {}))
    after_q_hist = cast(dict[str, int], after.get("q_evaluations_histogram", {}))
    delta_q_hist: dict[str, int] = {}
    delta_q_evaluations = 0
    for key, after_count in after_q_hist.items():
        diff = int(after_count) - int(before_q_hist.get(key, 0))
        if diff > 0:
            delta_q_hist[key] = diff
            delta_q_evaluations += int(key) * diff

    delta_early_stop = max(0, int(after.get("early_stop_count", 0)) - int(before.get("early_stop_count", 0)))
    return {
        "forward_calls": delta_calls,
        "total_reasoning_steps": delta_steps,
        "avg_reasoning_steps": (float(delta_steps) / delta_calls) if delta_calls else 0.0,
        "total_q_evaluations": delta_q_evaluations,
        "avg_q_evaluations": (float(delta_q_evaluations) / delta_calls) if delta_calls else 0.0,
        "early_stop_count": delta_early_stop,
        "early_stop_rate": (float(delta_early_stop) / delta_calls) if delta_calls else 0.0,
        "reasoning_steps_histogram": delta_hist,
        "q_evaluations_histogram": delta_q_hist,
    }


def append_runtime_stats_record(stats_path: str, payload: dict[str, Any]) -> None:
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def compact_text(value: str, max_length: int = 28) -> str:
    sanitized = value.replace("\n", "\\n")
    if len(sanitized) <= max_length:
        return sanitized
    return sanitized[: max_length - 3] + "..."


def supports_dynamic_ui() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def build_runtime_stats_table(stats: dict[str, Any], max_steps: int) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("Calls", str(int(stats["forward_calls"])))
    table.add_row("Avg loops", f"{float(stats['avg_reasoning_steps']):.2f}/{max_steps}")
    table.add_row("Avg q eval", f"{float(stats['avg_q_evaluations']):.2f}")
    table.add_row("Early stop", f"{float(stats['early_stop_rate']):.2%}")
    return table


def build_settings_table(model: LDSForCausalLM, args: argparse.Namespace) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold yellow")
    table.add_column(style="white")
    table.add_row("q-stop-threshold", f"{float(getattr(model, 'q_threshold', args.q_stop_threshold)):.2f}")
    table.add_row("n-recursion", str(int(model.N)))
    table.add_row("device", str(get_input_device(model)))
    return table


def apply_q_stop_threshold(model: LDSForCausalLM, args: argparse.Namespace, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("q-stop-threshold must be between 0.0 and 1.0")
    args.q_stop_threshold = value
    model.q_threshold = value
    model.config.q_threshold = value


def build_header_panel(model: LDSForCausalLM, args: argparse.Namespace) -> Panel:
    model_label = args.model_path.rstrip("/").split("/")[-1] or args.model_path
    header = Text()
    header.append(model_label, style="bold white")
    header.append("  |  ", style="dim")
    header.append(f"device={get_input_device(model)}", style="cyan")
    header.append("  ")
    header.append(f"dtype={next(model.parameters()).dtype}", style="magenta")
    header.append("  ")
    header.append(f"N={model.N}", style="green")
    header.append("  ")
    header.append(f"q={float(getattr(model, 'q_threshold', 0.0)):.2f}", style="yellow")
    return Panel(header, title="LDS Chat CLI", border_style="bright_blue")


def normalize_token_label(token_text: str, max_length: int = 12) -> str:
    if token_text == "":
        label = "<empty>"
    else:
        label = token_text.replace("\n", "<nl>").replace("\t", "<tab>")
        label = label.replace(" ", "<sp>") if label.strip() == "" else label
    if len(label) <= max_length:
        return label
    return label[: max_length - 3] + "..."


def loop_style(loop_count: int, max_loop_count: int) -> str:
    if max_loop_count <= 0:
        return "black on grey62"

    ratio = float(loop_count) / float(max_loop_count)
    if ratio <= 0.25:
        return "black on spring_green3"
    if ratio <= 0.5:
        return "black on khaki3"
    if ratio <= 0.75:
        return "white on dark_orange3"
    return "white on indian_red"


def build_token_loop_text(token_chips: list[TokenLoopChip], max_loop_count: int) -> Text:
    chips_text = Text()
    for chip in token_chips:
        chips_text.append(
            f" {normalize_token_label(chip.text)} {chip.loop_count} ",
            style=loop_style(chip.loop_count, max_loop_count),
        )
        chips_text.append(" ")
    return chips_text


def build_token_loop_legend(max_loop_count: int) -> Text:
    legend = Text("Loop intensity  ", style="dim")
    samples = [
        (max(1, int(max_loop_count * 0.2)), "low"),
        (max(1, int(max_loop_count * 0.45)), "mid"),
        (max(1, int(max_loop_count * 0.7)), "high"),
        (max(1, max_loop_count), "peak"),
    ]
    for loop_count, label in samples:
        legend.append(f" {label} ", style=loop_style(loop_count, max_loop_count))
        legend.append(" ")
    return legend


def build_token_loop_group(token_chips: list[TokenLoopChip], max_loop_count: int) -> Group:
    return Group(
        Text("Token loops", style="bold magenta"),
        build_token_loop_legend(max_loop_count),
        build_token_loop_text(token_chips, max_loop_count),
    )


def build_conversation_panel(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None,
    live_state: LiveGenerationState | None,
) -> Panel:
    renderables: list[Any] = []
    if system_prompt:
        system_text = Text()
        system_text.append("System\n", style="bold yellow")
        system_text.append(system_prompt.strip())
        renderables.append(system_text)

    if not messages and live_state is None:
        renderables.append(Text("Enter a prompt to see the response stream live.", style="dim"))

    for message in messages:
        message_text = Text()
        role = message["role"]
        if role == "user":
            message_text.append("You\n", style="bold cyan")
        else:
            message_text.append("Assistant\n", style="bold green")
        message_text.append(str(message["content"]).rstrip())
        renderables.append(message_text)

        token_chips = cast(list[TokenLoopChip] | None, message.get("token_chips"))
        token_chip_max = int(message.get("token_chip_max", 0))
        if role == "assistant" and token_chips:
            renderables.append(build_token_loop_group(token_chips, token_chip_max))

    if live_state is not None:
        live_text = Text()
        live_text.append("Assistant\n", style="bold green")
        if live_state.response:
            live_text.append(live_state.response)
        else:
            live_text.append("thinking...", style="dim")
        renderables.append(live_text)

        if live_state.token_chips:
            renderables.append(build_token_loop_group(live_state.token_chips, live_state.max_loop_count))

    conversation_body: Any
    if not renderables:
        conversation_body = Text("", style="white")
    else:
        conversation_body = Group(*renderables)

    return Panel(conversation_body, title="Conversation", border_style="cyan", padding=(1, 2))


def build_status_panel(
    model: LDSForCausalLM,
    args: argparse.Namespace,
    live_state: LiveGenerationState | None,
) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold magenta")
    table.add_column(style="white")

    phase = live_state.phase if live_state is not None else "Ready"
    token_count = live_state.token_count if live_state is not None else 0
    loop_count = live_state.current_loop_count if live_state is not None else 0
    loop_max = live_state.max_loop_count if live_state is not None else int(model.N)
    current_chunk = compact_text(live_state.current_chunk) if live_state is not None else "-"

    table.add_row("State", phase)
    table.add_row("Tokens", str(token_count))
    table.add_row("Current loop", f"{loop_count}/{loop_max}" if token_count else f"0/{loop_max}")
    table.add_row("Chunk", current_chunk or "-")
    table.add_row("Runtime log", "off" if args.no_runtime_stats else "jsonl")

    if not args.no_runtime_stats:
        stats = get_runtime_stats_snapshot(model)
        table.add_row("Session calls", str(int(stats["forward_calls"])))
        table.add_row("Session avg", f"{float(stats['avg_reasoning_steps']):.2f}/{model.N}")

    return Panel(table, title="Generation", border_style="magenta", padding=(1, 1), width=34)


def render_chat_cli(
    model: LDSForCausalLM,
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    live_state: LiveGenerationState | None = None,
) -> Group:
    footer = Text(
        "Commands: /reset  /stats  /settings  /set q-stop-threshold <value>  /exit  |  During generation: ESC to stop",
        style="dim",
    )
    return Group(
        build_header_panel(model, args),
        Columns(
            [
                build_conversation_panel(messages, system_prompt=args.system_prompt, live_state=live_state),
                build_status_panel(model, args, live_state),
            ],
            expand=True,
            equal=False,
        ),
        footer,
    )


def redraw_chat_screen(
    console: Console,
    model: LDSForCausalLM,
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    live_state: LiveGenerationState | None = None,
) -> None:
    if supports_dynamic_ui():
        console.clear()
    console.print(render_chat_cli(model, args, messages, live_state=live_state))


def show_runtime_stats_panel(console: Console, model: LDSForCausalLM) -> None:
    stats = get_runtime_stats_snapshot(model)
    console.print(Panel(build_runtime_stats_table(stats, model.N), title="Session Stats", border_style="green"))


def show_settings_panel(console: Console, model: LDSForCausalLM, args: argparse.Namespace) -> None:
    console.print(Panel(build_settings_table(model, args), title="Settings", border_style="yellow"))


def maybe_report_runtime_stats(
    model: LDSForCausalLM,
    *,
    before_stats: dict[str, Any] | None,
    args: argparse.Namespace,
    prompt: str,
    response: str,
) -> None:
    if before_stats is None:
        return

    after_stats = get_runtime_stats_snapshot(model)
    turn_stats = diff_runtime_stats(before_stats, after_stats)

    append_runtime_stats_record(
        args.runtime_stats_path,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_path": args.model_path,
            "device": str(get_input_device(model)),
            "dtype": str(next(model.parameters()).dtype),
            "N": int(model.N),
            "q_threshold": float(getattr(model, "q_threshold", 0.0)),
            "prompt": prompt,
            "response": response,
            "turn_stats": turn_stats,
            "session_stats": after_stats,
        },
    )


def build_prompt_text(
    tokenizer,
    messages: list[dict[str, str]],
    system_prompt: str | None,
    use_chat_template: bool,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        prompt_messages: list[dict[str, str]] = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.extend(messages)
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    transcript: list[str] = []
    if system_prompt:
        transcript.append(f"System: {system_prompt}")
    for message in messages:
        role = message["role"].capitalize()
        transcript.append(f"{role}: {message['content']}")
    transcript.append("Assistant:")
    return "\n".join(transcript)


def trim_history(messages: list[dict[str, str]], max_history_turns: int) -> list[dict[str, str]]:
    if max_history_turns <= 0:
        return messages
    max_messages = max_history_turns * 2
    return messages[-max_messages:]


@torch.inference_mode()
def generate_once(
    model: LDSForCausalLM,
    tokenizer,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    stop_callback: Callable[[], bool] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    prompt_text = build_prompt_text(
        tokenizer,
        messages=trim_history(messages, args.max_history_turns),
        system_prompt=args.system_prompt,
        use_chat_template=not args.disable_chat_template,
    )

    device = get_input_device(model)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    before_stats = None if args.no_runtime_stats else get_runtime_stats_snapshot(model)

    streamed_token_ids: list[int] = []
    streamed_text = ""

    def handle_generation_step(payload: dict[str, Any]) -> None:
        nonlocal streamed_text
        if progress_callback is None:
            return

        token_ids = [int(token_id) for token_id in cast(list[int], payload.get("token_ids", []))]
        if not token_ids:
            return

        streamed_token_ids.extend(token_ids[:1])
        decoded_text = tokenizer.decode(streamed_token_ids, skip_special_tokens=True)
        delta_text = decoded_text[len(streamed_text):] if decoded_text.startswith(streamed_text) else decoded_text
        streamed_text = decoded_text
        progress_callback(
            {
                **payload,
                "delta_text": delta_text,
                "decoded_text": decoded_text,
                "current_chunk": tokenizer.decode(token_ids[:1], skip_special_tokens=False),
            }
        )

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        step_callback=handle_generation_step,
        stop_callback=stop_callback,
    )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), before_stats


def run_live_generation(
    model: LDSForCausalLM,
    tokenizer,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    console: Console,
) -> tuple[str, dict[str, Any] | None, list[TokenLoopChip]]:
    if not supports_dynamic_ui():
        response, before_stats = generate_once(model, tokenizer, messages, args)
        return response, before_stats, []

    live_state = LiveGenerationState(max_loop_count=int(model.N), phase="Thinking")
    rendered_response = ""
    esc_watcher = EscapeKeyWatcher()
    esc_watcher.start()

    try:
        with Live(
            render_chat_cli(model, args, messages, live_state=live_state),
            console=console,
            auto_refresh=False,
            transient=True,
        ) as live:

            def on_progress(payload: dict[str, Any]) -> None:
                nonlocal rendered_response
                reasoning_steps = cast(list[int] | None, payload.get("reasoning_steps"))
                if reasoning_steps:
                    live_state.current_loop_count = int(reasoning_steps[0])
                live_state.max_loop_count = int(payload.get("max_reasoning_steps", model.N))
                live_state.current_chunk = str(payload.get("current_chunk", ""))
                live_state.token_count = int(payload.get("step_index", live_state.token_count))
                live_state.phase = "Stopping" if esc_watcher.aborted() else "Generating"

                delta_text = str(payload.get("delta_text", ""))
                if delta_text:
                    live_state.token_chips.append(
                        TokenLoopChip(
                            text=delta_text,
                            loop_count=live_state.current_loop_count,
                        )
                    )

                if not delta_text:
                    live.update(render_chat_cli(model, args, messages, live_state=live_state), refresh=True)
                    return

                for char in delta_text:
                    rendered_response += char
                    live_state.response = rendered_response
                    live.update(render_chat_cli(model, args, messages, live_state=live_state), refresh=True)

            response, before_stats = generate_once(
                model,
                tokenizer,
                messages,
                args,
                progress_callback=on_progress,
                stop_callback=esc_watcher.aborted,
            )
            live_state.response = rendered_response or response
            live_state.current_chunk = ""
            live_state.aborted = esc_watcher.aborted()
            live_state.phase = "Stopped" if live_state.aborted else "Done"
            live.update(render_chat_cli(model, args, messages, live_state=live_state), refresh=True)
    finally:
        esc_watcher.stop()

    return live_state.response or response, before_stats, list(live_state.token_chips)


def run_interactive(model: LDSForCausalLM, tokenizer, args: argparse.Namespace) -> None:
    console = Console()
    history: list[dict[str, Any]] = []
    redraw_chat_screen(console, model, args, history)
    while True:
        try:
            prompt = console.input("\n[bold cyan]You> [/]").strip()
        except EOFError:
            console.print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if prompt.lower() == "/reset":
            history.clear()
            if not args.no_runtime_stats:
                model.reset_runtime_stats()
            redraw_chat_screen(console, model, args, history)
            continue
        if prompt.lower() == "/stats":
            if args.no_runtime_stats:
                console.print(Panel("Runtime stats are disabled.", border_style="yellow"))
            else:
                show_runtime_stats_panel(console, model)
            continue
        if prompt.lower() == "/settings":
            show_settings_panel(console, model, args)
            continue
        if prompt.lower().startswith("/set "):
            parts = prompt.split(maxsplit=2)
            if len(parts) != 3:
                console.print(
                    Panel(
                        "Usage: /set q-stop-threshold <value>",
                        title="Invalid Setting Command",
                        border_style="red",
                    )
                )
                continue

            setting_name = parts[1].strip().lower()
            raw_value = parts[2].strip()
            if setting_name not in {"q-stop-threshold", "q_threshold", "q"}:
                console.print(
                    Panel(
                        f"Unsupported setting: {setting_name}",
                        title="Invalid Setting",
                        border_style="red",
                    )
                )
                continue

            try:
                apply_q_stop_threshold(model, args, float(raw_value))
            except ValueError as error:
                console.print(Panel(str(error), title="Invalid Value", border_style="red"))
                continue

            redraw_chat_screen(console, model, args, history)
            console.print(
                Panel(
                    f"q-stop-threshold updated to {args.q_stop_threshold:.2f}",
                    title="Setting Updated",
                    border_style="green",
                )
            )
            continue

        history.append({"role": "user", "content": prompt})
        response, before_stats, token_chips = run_live_generation(model, tokenizer, history, args, console)
        history.append(
            {
                "role": "assistant",
                "content": response,
                "token_chips": token_chips,
                "token_chip_max": int(model.N),
            }
        )
        maybe_report_runtime_stats(
            model,
            before_stats=before_stats,
            args=args,
            prompt=prompt,
            response=response,
        )
        redraw_chat_screen(console, model, args, history)


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args)

    if args.interactive or args.prompt is None:
        run_interactive(model, tokenizer, args)
        return

    console = Console()
    messages = [{"role": "user", "content": args.prompt}]
    response, before_stats, token_chips = run_live_generation(
        model,
        tokenizer,
        messages,
        args,
        console,
    )
    if supports_dynamic_ui():
        redraw_chat_screen(
            console,
            model,
            args,
            messages
            + [{"role": "assistant", "content": response, "token_chips": token_chips, "token_chip_max": int(model.N)}],
        )
    else:
        print(response)
    maybe_report_runtime_stats(
        model,
        before_stats=before_stats,
        args=args,
        prompt=args.prompt,
        response=response,
    )


if __name__ == "__main__":
    main()