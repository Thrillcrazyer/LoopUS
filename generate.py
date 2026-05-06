"""Generate qualitative LDS samples from a saved model or checkpoint."""

from __future__ import annotations

import argparse
import os

from utils.inference import load_lds_model, resolve_device_and_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text samples from an LDS model")
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B"),
    )
    parser.add_argument(
        "--decomposed-model",
        "--from-hub",
        dest="decomposed_model",
        type=str,
        default=None,
        help="HF Hub repo ID or local save_pretrained directory for a trained LDS model",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/step_5000",
        help="Legacy checkpoint directory containing combined_model.pt",
    )
    parser.add_argument(
        "--encoder-layers",
        type=str,
        default="none",
        help="Encoder layers or 'none' for zero encoder blocks",
    )
    parser.add_argument(
        "--decoder-layers",
        type=str,
        default=None,
        help="Decoder layers or 'none' for zero decoder blocks",
    )
    parser.add_argument(
        "--n-recursion",
        type=int,
        default=8,
        help="Number of reasoning passes per token generation step",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt; if omitted, a small qualitative prompt set is used",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the model on (e.g., 'cpu', 'cuda', 'cuda:0', 'auto')",
    )
    return parser.parse_args()


def _default_prompts() -> list[str]:
    return [
        "The meaning of life is",
        "In a distant galaxy, scientists discovered",
        "Machine learning is a field of",
        "Once upon a time, there was a",
    ]


def main() -> None:
    args = parse_args()
    load_device, adapter_device, dtype = resolve_device_and_dtype(args.device)
    combined = load_lds_model(
        model_name=args.model_name,
        device=load_device,
        dtype=dtype,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        decomposed_model=args.decomposed_model,
        checkpoint_dir=args.checkpoint_dir,
        n_recursion=args.n_recursion,
    )

    target_device = adapter_device if adapter_device != "auto" else "cuda" if dtype != combined.dtype else next(combined.parameters()).device
    if target_device != "auto":
        combined = combined.to(device=target_device, dtype=dtype)

    tokenizer = combined.tokenizer
    prompts = [args.prompt] if args.prompt else _default_prompts()

    print(f"Model ready on {next(combined.parameters()).device} | N={combined.N}")

    print("\n" + "=" * 70)
    print(f"TOP-K SAMPLING (temperature={args.temperature}, top_k={args.top_k})")
    print("=" * 70)
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(combined.parameters()).device)
        output = combined.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"[Prompt] {prompt}")
        print(f"[Output] {tokenizer.decode(output[0], skip_special_tokens=True)}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("GREEDY DECODING")
    print("=" * 70)
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(combined.parameters()).device)
        output = combined.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        print(f"[Prompt] {prompt}")
        print(f"[Output] {tokenizer.decode(output[0], skip_special_tokens=True)}")
        print("-" * 70)

    test_prompt = prompts[0]
    print("\n" + "=" * 70)
    print(f"RECURSION DEPTH COMPARISON | Prompt: {test_prompt}")
    print("=" * 70)
    for n_recursion in [1, 2, 4, 8]:
        combined.N = n_recursion
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(next(combined.parameters()).device)
        output = combined.generate(input_ids, max_new_tokens=50, do_sample=False)
        print(f"[N={n_recursion:2d}] {tokenizer.decode(output[0], skip_special_tokens=True)}")

    combined.N = args.n_recursion
    print("=" * 70)


if __name__ == "__main__":
    main()