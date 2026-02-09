#!/bin/bash
set -euo pipefail

DEFAULT_GREC_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec"
: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

cd "$GREC_ROOT" || exit 1

# IAG: Instruments + Arts + Games
: "${ROOT_DIR:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${USE_MULTI_DATASETS:=true}"
: "${DATASETS:=Instruments Arts Games}"
: "${INDEX_RUN_SCRIPT_DIR:=Instruments-Arts-Games-qwen3-embedding-4B-rq4_cb512-512-512-512_sk0.0-0.0-0.0-0.003}"

export ROOT_DIR MODEL_NAME USE_MULTI_DATASETS DATASETS INDEX_RUN_SCRIPT_DIR

# Optional overrides (examples):
#   NPROC_PER_NODE=4 BATCH_SIZE=128 EPOCHS=1000 INDEX_CODEBOOK_SIZE=512
bash "$GREC_ROOT/scripts/index/base/train.sh"
