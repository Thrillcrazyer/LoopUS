#!/bin/bash
# Shared download/training settings. Environment variables can override these.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
TRAIN_DATASET="${TRAIN_DATASET:-HuggingFaceFW/fineweb-edu}"
TRAIN_CONFIG="${TRAIN_CONFIG-CC-MAIN-2025-26}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TRAIN_MAX_TOKENS="${TRAIN_MAX_TOKENS:-3000000000}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-200000}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

DOWNLOAD_DIR="${DOWNLOAD_DIR:-$PROJECT_ROOT/downloads}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$DOWNLOAD_DIR/models/${MODEL_NAME//\//--}}"
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-$DOWNLOAD_DIR/datasets/${TRAIN_DATASET//\//--}/${TRAIN_CONFIG:-default}/${TRAIN_SPLIT}}"

DOWNLOAD_ARGS=(
    --model-name "$MODEL_NAME"
    --model-dir "$LOCAL_MODEL_DIR"
    --train-dataset "$TRAIN_DATASET"
    --train-config "$TRAIN_CONFIG"
    --train-split "$TRAIN_SPLIT"
    --train-max-tokens "$TRAIN_MAX_TOKENS"
    --train-max-samples "$TRAIN_MAX_SAMPLES"
    --max-length "$MAX_LENGTH"
    --dataset-dir "$LOCAL_DATASET_DIR"
)
