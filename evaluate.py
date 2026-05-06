"""
Evaluate LDSForCausalLM using lm-evaluation-harness.

This entrypoint is LDS-only. Baseline evaluation for the original pretrained
model lives in ``test_pretrain_model.py``.

Usage:
    # Evaluate a decomposed model from legacy checkpoint weights
    python evaluate.py --model-name Qwen/Qwen3-1.7B \
        --checkpoint-dir checkpoints/step_18000

    # Evaluate on specific tasks
    python evaluate.py --model-name Qwen/Qwen3-1.7B --tasks mmlu,hellaswag,arc_easy

    # Quick evaluation with limited samples
    python evaluate.py --model-name Qwen/Qwen3-1.7B --limit 100

    # Evaluate a saved LDS model from HF Hub
    python evaluate.py --decomposed-model your-name/lds-qwen3-step20000

    # Override recursion depth at evaluation time
    python evaluate.py --decomposed-model your-name/lds-qwen3-step20000 \
        --n-recursion 3
"""

import argparse
import json
import os
from contextlib import nullcontext

import torch
from utils.common import SEED, set_seed
from utils.inference import load_lds_model, resolve_device_and_dtype
from utils.lm_eval_harness import (
    DDDModelAdapter,
    DEFAULT_TASKS,
    run_lm_eval,
    format_results_log,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model using lm-evaluation-harness")
    p.add_argument("--model-name", type=str, default=os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B"))
    p.add_argument("--mode", type=str, default="decomposed", choices=["decomposed"],
                   help=argparse.SUPPRESS)
    p.add_argument("--decomposed-model", "--from-hub", dest="decomposed_model", type=str, default=None,
                   help="HF Hub repo ID or local save_pretrained directory for a trained LDS model")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Legacy checkpoint directory containing combined_model.pt")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated list of lm-eval tasks "
                        "(default: mmlu,hellaswag,arc_easy,arc_challenge,piqa,"
                        "winogrande,lambada_openai,wikitext,boolq,openbookqa)")
    p.add_argument("--num-fewshot", type=int, default=None,
                   help="Number of few-shot examples (None = use task defaults)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit samples per task (for quick testing)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024,
                   help="Maximum sequence length for the model")
    p.add_argument("--n-recursion", type=int, default=8,
                   help="N: number of reasoning passes per recursion step (LDSForCausalLM.N)")
    p.add_argument("--q-eval-interval", type=int, default=1,
                   help="Evaluate the q-stop head every k reasoning passes (default: 1 = every pass)")
    p.add_argument("--encoder-layers", type=str, default=None,
                   help="Encoder layers or 'none' for zero encoder blocks")
    p.add_argument("--decoder-layers", type=str, default=None,
                   help="Decoder layers or 'none' for zero decoder blocks")
    p.add_argument("--q-stop-threshold", type=float, default=None,
                   help="Override the LDS early-stop threshold used during evaluation")
    p.add_argument("--q-stop-mode", type=str, default="all", choices=["all", "any"],
                   help="Compatibility flag; evaluation currently uses threshold-only stopping")
    p.add_argument("--output-json", type=str, default=None,
                   help="Path to write detailed results JSON")
    p.add_argument("--log-samples", action="store_true",
                   help="Log individual sample results from lm-eval")
    p.add_argument("--bootstrap-iters", type=int, default=0,
                   help="Bootstrap iterations for lm-eval stderr computation (default: 0 for speed)")
    p.add_argument("--cache-requests", action="store_true",
                   help="Cache built lm-eval requests to speed up repeated runs")
    p.add_argument("--rewrite-requests-cache", action="store_true",
                   help="Refresh the lm-eval request cache before evaluating")
    p.add_argument("--verbosity", type=str, default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="lm-eval logging verbosity")
    p.add_argument("--runtime-stats", action="store_true",
                   help="Collect LDS reasoning runtime statistics (disabled by default for speed)")
    p.add_argument("--halting-strategy", type=str, default="threshold",
                   choices=["threshold", "convergence", "cdf"],
                   help="Halting strategy: threshold (q>th), convergence (|h_t+1-h_t|<eps), cdf (cumulative hazard)")
    p.add_argument("--convergence-epsilon", type=float, default=1e-2,
                   help="Epsilon for convergence halting strategy")
    p.add_argument("--device", type=str, default="auto",
                   help="Device to run the model on (e.g., 'cpu', 'cuda', 'auto')")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.q_eval_interval < 1:
        raise ValueError("--q-eval-interval must be >= 1")
    set_seed(args.seed)
    load_device, adapter_device, dtype = resolve_device_and_dtype(args.device)

    # --- Determine tasks ---
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = list(DEFAULT_TASKS)

    # --- Load model & create adapter ---
    combined = load_lds_model(
        model_name=args.model_name,
        device=load_device,
        dtype=dtype,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        decomposed_model=args.decomposed_model,
        checkpoint_dir=args.checkpoint_dir,
        n_recursion=args.n_recursion,
        q_stop_threshold=args.q_stop_threshold,
        q_eval_interval=args.q_eval_interval,
        halting_strategy=args.halting_strategy,
        convergence_epsilon=args.convergence_epsilon,
    )
    combined.set_runtime_stats_enabled(args.runtime_stats)
    if args.runtime_stats:
        combined.reset_runtime_stats()
    adapter = DDDModelAdapter(
        combined_model=combined,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=adapter_device,
    )

    # --- Run evaluation ---
    print(f"\nRunning lm-evaluation-harness...")
    print(f"  Tasks: {tasks}")
    print("  Mode: decomposed")
    print(f"  N (plateau): {args.n_recursion}")
    print(f"  Q eval interval: {args.q_eval_interval}")
    if args.decomposed_model:
        print(f"  Saved model: {args.decomposed_model}")
    if args.checkpoint_dir:
        print(f"  Legacy checkpoint: {args.checkpoint_dir}")
    if args.q_stop_threshold is not None:
        print(f"  Q-stop threshold: {args.q_stop_threshold}")
    if args.limit:
        print(f"  Limit: {args.limit} samples/task")
    print(f"  Bootstrap iters: {args.bootstrap_iters}")
    if args.cache_requests:
        print("  Request cache: enabled")
    if args.rewrite_requests_cache:
        print("  Request cache rewrite: enabled")
    if args.num_fewshot is not None:
        print(f"  Few-shot: {args.num_fewshot}")
    print(f"  Runtime stats: {'enabled' if args.runtime_stats else 'disabled'}")
    print()

    eval_context = torch.inference_mode if hasattr(torch, "inference_mode") else nullcontext
    with eval_context():
        results = run_lm_eval(
            model=adapter,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=args.log_samples,
            verbosity=args.verbosity,
            bootstrap_iters=args.bootstrap_iters,
            cache_requests=args.cache_requests,
            rewrite_requests_cache=args.rewrite_requests_cache,
        )
    recursion_stats = None
    if args.runtime_stats:
        recursion_stats = combined.get_runtime_stats()

    # --- Print results ---
    print("\n" + "=" * 70)
    print(f"Model      : {args.decomposed_model or args.model_name}")
    print("Mode       : decomposed")
    print(f"N (plateau): {args.n_recursion}")
    print(f"Q eval int.: {args.q_eval_interval}")
    print(f"Halting    : {args.halting_strategy}")
    if args.halting_strategy == "convergence":
        print(f"Conv. eps  : {args.convergence_epsilon}")
    if args.checkpoint_dir:
        print(f"Checkpoint : {args.checkpoint_dir}")
    if args.q_stop_threshold is not None:
        print(f"Q-stop th. : {args.q_stop_threshold}")
    print(f"Tasks      : {', '.join(tasks)}")
    print("=" * 70)

    log_str = format_results_log(results, prefix="  ")
    print(log_str)
    if recursion_stats is not None:
        print("  Recursion stats:")
        print(f"    forward calls     : {recursion_stats['forward_calls']}")
        print(f"    avg steps         : {recursion_stats['avg_reasoning_steps']:.2f}/{args.n_recursion}")
        print(f"    avg q evals       : {recursion_stats['avg_q_evaluations']:.2f}")
        print(f"    early-stop rate   : {recursion_stats['early_stop_rate']:.2%}")
        mean_final_q_min = recursion_stats.get("mean_final_q_min")
        if mean_final_q_min is not None:
            print(f"    mean final q-min  : {mean_final_q_min:.4f}")
        print(f"    step histogram    : {recursion_stats['reasoning_steps_histogram']}")
        print(f"    q-eval histogram  : {recursion_stats['q_evaluations_histogram']}")
    print("=" * 70)

    # --- Save results ---
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        output = {
            "model_name": args.model_name,
            "decomposed_model": args.decomposed_model,
            "mode": "decomposed",
            "tasks": tasks,
            "n_recursion": args.n_recursion,
            "q_eval_interval": args.q_eval_interval,
            "halting_strategy": args.halting_strategy,
            "convergence_epsilon": args.convergence_epsilon,
            "checkpoint_dir": args.checkpoint_dir,
            "q_stop_threshold": args.q_stop_threshold,
            "q_stop_mode": args.q_stop_mode,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "seed": args.seed,
            "bootstrap_iters": args.bootstrap_iters,
            "cache_requests": args.cache_requests,
            "rewrite_requests_cache": args.rewrite_requests_cache,
            "runtime_stats_enabled": args.runtime_stats,
            "recursion_stats": recursion_stats,
            **results,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {args.output_json}")


if __name__ == "__main__":
    main()
