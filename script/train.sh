#!/bin/bash
set -euo pipefail

# Reproducible public training template for DDD.

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1

MODEL_NAME="Qwen/Qwen3-1.7B"
ENCODER_LAYERS="0..1"
DECODER_LAYERS="27..27"

TRAIN_DATASET="HuggingFaceFW/fineweb-edu"
TRAIN_CONFIG="CC-MAIN-2025-26"
TRAIN_SPLIT="train"
TRAIN_MAX_TOKENS=1500000000

EPOCHS=1
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=64
LEARNING_RATE=5e-5
MAX_LENGTH=1024
SEED=2026

N_SUPERVISION=5
N_REASONING_STEPS=20

VAL_RATIO=0.0001
EVAL_INTERVAL=-1
LOG_INTERVAL=50
EVAL_JSONL_PATH="logs/eval_metrics.jsonl"
TRAIN_JSONL_PATH="logs/train_metrics_$(date +%Y%m%d_%H%M%S).jsonl"

Q_STOP_THRESHOLD=0.55
Q_STOP_MODE="all"
MIXED_PRECISION="bf16"

SAVE_INTERVAL=5000
MAX_CHECKPOINTS=3
CHECKPOINT_DIR="checkpoints/example_run"
RESUME=""
FROM_HUB=""

LM_EVAL_AT_SAVE=true
LM_EVAL_TASKS="wikitext"
LM_EVAL_LIMIT=200
LM_EVAL_BATCH_SIZE=4
LM_EVAL_MAX_LENGTH=1024
LM_EVAL_NUM_FEWSHOT=-1

EVAL_RESPONSES_DIR="logs/eval_responses"
EVAL_MAX_RESPONSES=100

WARMUP_STEPS=300
LR_SCHEDULER="cosine"

NUM_WORKERS=8
PIN_MEMORY=true
USE_ZERO3=false

WANDB_ENABLED=false
WANDB_PROJECT="LoopUS"
WANDB_RUN_NAME="LoopUS_example_$(date +%Y%m%d_%H%M)"
WANDB_ENTITY=""

HUB_REPO_ID=""
HUB_PUSH_INTERVAL=0
HUB_PRIVATE=false

ACCEL_CONFIG=""
if [[ "$USE_ZERO3" == "true" ]]; then
    ACCEL_CONFIG="ds_configs/accelerate_ds_zero3.yaml"
fi

CMD=(uv run accelerate launch)
[[ -n "$ACCEL_CONFIG" ]] && CMD+=(--config_file "$ACCEL_CONFIG")
CMD+=(
    train.py
    --seed "$SEED"
    --model-name "$MODEL_NAME"
    --train-dataset "$TRAIN_DATASET"
    --train-config "$TRAIN_CONFIG"
    --train-split "$TRAIN_SPLIT"
    --train-max-tokens "$TRAIN_MAX_TOKENS"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --max-length "$MAX_LENGTH"
    --n-supervision "$N_SUPERVISION"
    --n-reasoning-steps "$N_REASONING_STEPS"
    --val-ratio "$VAL_RATIO"
    --eval-interval "$EVAL_INTERVAL"
    --log-interval "$LOG_INTERVAL"
    --eval-jsonl-path "$EVAL_JSONL_PATH"
    --train-jsonl-path "$TRAIN_JSONL_PATH"
    --q-stop-threshold "$Q_STOP_THRESHOLD"
    --q-stop-mode "$Q_STOP_MODE"
    --save-interval "$SAVE_INTERVAL"
    --max-checkpoints "$MAX_CHECKPOINTS"
    --mixed-precision "$MIXED_PRECISION"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --encoder-layers "$ENCODER_LAYERS"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --eval-responses-dir "$EVAL_RESPONSES_DIR"
    --eval-max-responses "$EVAL_MAX_RESPONSES"
    --warmup-steps "$WARMUP_STEPS"
    --lr-scheduler "$LR_SCHEDULER"
    --num-workers "$NUM_WORKERS"
)

[[ -n "$DECODER_LAYERS" ]] && CMD+=(--decoder-layers "$DECODER_LAYERS")
[[ -n "$RESUME" ]] && CMD+=(--resume "$RESUME")
[[ -n "$FROM_HUB" ]] && CMD+=(--from-hub "$FROM_HUB")
[[ "$PIN_MEMORY" == "true" ]] && CMD+=(--pin-memory)

if [[ "$WANDB_ENABLED" == "true" ]]; then
    CMD+=(--wandb --wandb-project "$WANDB_PROJECT")
    [[ -n "$WANDB_RUN_NAME" ]] && CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
    [[ -n "$WANDB_ENTITY" ]] && CMD+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ "$HUB_PUSH_INTERVAL" -gt 0 ]] && [[ -n "$HUB_REPO_ID" ]]; then
    CMD+=(--hub-push-interval "$HUB_PUSH_INTERVAL" --hub-repo-id "$HUB_REPO_ID")
    [[ "$HUB_PRIVATE" == "true" ]] && CMD+=(--hub-private)
fi

if [[ "$LM_EVAL_AT_SAVE" == "true" ]]; then
    CMD+=(
        --lm-eval-at-save
        --lm-eval-tasks "$LM_EVAL_TASKS"
        --lm-eval-limit "$LM_EVAL_LIMIT"
        --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
        --lm-eval-max-length "$LM_EVAL_MAX_LENGTH"
        --lm-eval-num-fewshot "$LM_EVAL_NUM_FEWSHOT"
    )
fi

mkdir -p logs
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

echo ">>> ${CMD[*]}"
echo ">>> Log file: ${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"