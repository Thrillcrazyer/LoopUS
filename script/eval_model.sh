#!/bin/bash
set -euo pipefail

# Reproducible public evaluation template for DDD.

MODEL_NAME="Qwen/Qwen3-1.7B"
DECOMPOSED_MODEL=""
CHECKPOINT_DIR=""
TASKS="mmlu,hellaswag,arc_easy,arc_challenge,piqa,winogrande"
N_RECURSION=8
BATCH_SIZE=8
MAX_LENGTH=1024
Q_STOP_THRESHOLD=0.6
DEVICE="auto"
SEED=2026
OUTPUT_JSON="results/eval.json"

mkdir -p "$(dirname "$OUTPUT_JSON")"

CMD=(
    uv run evaluate.py
    --model-name "$MODEL_NAME"
    --tasks "$TASKS"
    --n-recursion "$N_RECURSION"
    --batch-size "$BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --device "$DEVICE"
    --seed "$SEED"
    --output-json "$OUTPUT_JSON"
)

[[ -n "$DECOMPOSED_MODEL" ]] && CMD+=(--decomposed-model "$DECOMPOSED_MODEL")
[[ -n "$CHECKPOINT_DIR" ]] && CMD+=(--checkpoint-dir "$CHECKPOINT_DIR")
[[ -n "$Q_STOP_THRESHOLD" ]] && CMD+=(--q-stop-threshold "$Q_STOP_THRESHOLD")

echo ">>> ${CMD[*]}"
"${CMD[@]}"
