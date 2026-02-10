#!/usr/bin/env bash
set -euo pipefail

DEFAULT_GREC_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec"
DEFAULT_ROOT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian"

: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"
: "${ROOT_DIR:=$DEFAULT_ROOT_DIR}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/finetune/smoke_train.sh [train_script_path]

Examples:
  # Use default Instruments cb64 bundle script
  bash scripts/finetune/smoke_train.sh

  # Use another bundle script
  bash scripts/finetune/smoke_train.sh \
    scripts/finetune/Instruments_Instruments-qwen3-embedding-4B-rq4_cb128-128-128-128_sk0.0-0.0-0.0-0.003/train.sh

Overrides (env vars):
  DATASET, DATA_PATH, TASKS, TRAIN_PROMPT_SAMPLE_NUM, TRAIN_DATA_SAMPLE_NUM,
  EPOCHS, PER_DEVICE_BATCH_SIZE, GRAD_ACC, NUM_WORKERS,
  GPUS, NPROC, MASTER_PORT,
  SAVE_AND_EVAL_STRATEGY, EVAL_BY_DATASET, EVAL_MAIN_DATASET,
  REPORT_TO, WANDB_MODE, OUTPUT_DIR, TARGET_TRAIN_SCRIPT

Notes:
  - This script runs in debug mode (foreground, single process by default).
  - Goal is smoke test: quickly validate train/eval/save pipeline.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

DEFAULT_TARGET_SCRIPT="$GREC_ROOT/scripts/finetune/Instruments_Instruments-qwen3-embedding-4B-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003/train.sh"
: "${TARGET_TRAIN_SCRIPT:=$DEFAULT_TARGET_SCRIPT}"

if [[ $# -gt 0 ]]; then
  TARGET_TRAIN_SCRIPT="$1"
  shift
fi

if [[ "$TARGET_TRAIN_SCRIPT" != /* ]]; then
  if [[ -f "$TARGET_TRAIN_SCRIPT" ]]; then
    TARGET_TRAIN_SCRIPT="$(cd "$(dirname "$TARGET_TRAIN_SCRIPT")" && pwd)/$(basename "$TARGET_TRAIN_SCRIPT")"
  elif [[ -f "$GREC_ROOT/$TARGET_TRAIN_SCRIPT" ]]; then
    TARGET_TRAIN_SCRIPT="$GREC_ROOT/$TARGET_TRAIN_SCRIPT"
  else
    echo "Error: train script not found: $TARGET_TRAIN_SCRIPT" >&2
    exit 1
  fi
fi

if [[ ! -f "$TARGET_TRAIN_SCRIPT" ]]; then
  echo "Error: train script does not exist: $TARGET_TRAIN_SCRIPT" >&2
  exit 1
fi

: "${DATASET:=Instruments}"
: "${DATA_PATH:=${ROOT_DIR}/data}"
: "${TASKS:=item2index}"
: "${TRAIN_PROMPT_SAMPLE_NUM:=1}"
: "${TRAIN_DATA_SAMPLE_NUM:=16}"
: "${EPOCHS:=1}"
: "${PER_DEVICE_BATCH_SIZE:=2}"
: "${GRAD_ACC:=1}"
: "${NUM_WORKERS:=2}"
: "${GPUS:=0}"
: "${NPROC:=1}"
: "${MASTER_PORT:=33391}"
: "${SAVE_AND_EVAL_STRATEGY:=epoch}"
: "${EVAL_BY_DATASET:=true}"
: "${EVAL_MAIN_DATASET:=$DATASET}"
: "${REPORT_TO:=none}"
: "${WANDB_MODE:=disabled}"

smoke_ts="$(date +%Y%m%d_%H%M%S)"
: "${OUTPUT_DIR:=${ROOT_DIR}/ckpt/${DATASET}/smoke_sft_${smoke_ts}}"

export DATASET DATA_PATH TASKS
export TRAIN_PROMPT_SAMPLE_NUM TRAIN_DATA_SAMPLE_NUM
export EPOCHS PER_DEVICE_BATCH_SIZE GRAD_ACC NUM_WORKERS
export GPUS NPROC MASTER_PORT
export SAVE_AND_EVAL_STRATEGY EVAL_BY_DATASET EVAL_MAIN_DATASET
export REPORT_TO WANDB_MODE OUTPUT_DIR

echo "[smoke] GREC_ROOT=$GREC_ROOT"
echo "[smoke] TARGET_TRAIN_SCRIPT=$TARGET_TRAIN_SCRIPT"
echo "[smoke] DATASET=$DATASET TASKS=$TASKS"
echo "[smoke] TRAIN_DATA_SAMPLE_NUM=$TRAIN_DATA_SAMPLE_NUM"
echo "[smoke] EPOCHS=$EPOCHS BATCH=$PER_DEVICE_BATCH_SIZE GRAD_ACC=$GRAD_ACC"
echo "[smoke] EVAL_BY_DATASET=$EVAL_BY_DATASET EVAL_MAIN_DATASET=$EVAL_MAIN_DATASET"
echo "[smoke] OUTPUT_DIR=$OUTPUT_DIR"

cd "$GREC_ROOT" || exit 1

bash "$TARGET_TRAIN_SCRIPT" --debug "$@"

