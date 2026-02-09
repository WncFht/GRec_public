#!/bin/bash
set -euo pipefail

DEFAULT_GREC_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec"
: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

cd "$GREC_ROOT" || exit 1

# Single dataset: Instruments
: "${ROOT_DIR:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${USE_MULTI_DATASETS:=false}"
: "${DATASET:=Instruments}"

# RQ config: 4 layers, 32 codebook
: "${INDEX_N_LAYERS:=4}"
: "${INDEX_CODEBOOK_SIZE:=32}"
: "${INDEX_LAST_SK_EPSILON:=0.003}"
: "${INDEX_RUN_SCRIPT_DIR:=Instruments-qwen3-embedding-4B-rq4_cb32-32-32-32_sk0.0-0.0-0.0-0.003}"

export ROOT_DIR MODEL_NAME USE_MULTI_DATASETS DATASET INDEX_RUN_SCRIPT_DIR
export INDEX_N_LAYERS INDEX_CODEBOOK_SIZE INDEX_LAST_SK_EPSILON

# Optional overrides (examples):
#   NPROC_PER_NODE=4 BATCH_SIZE=256 EPOCHS=500
bash "$GREC_ROOT/scripts/index/base/train.sh"
