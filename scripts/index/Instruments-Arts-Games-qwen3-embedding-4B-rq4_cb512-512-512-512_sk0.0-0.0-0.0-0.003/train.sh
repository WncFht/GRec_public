#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# IAG: Instruments + Arts + Games
: "${ROOT_DIR:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${USE_MULTI_DATASETS:=true}"
: "${DATASETS:=Instruments Arts Games}"

export ROOT_DIR MODEL_NAME USE_MULTI_DATASETS DATASETS

# Optional overrides (examples):
#   NPROC_PER_NODE=4 BATCH_SIZE=128 EPOCHS=1000 INDEX_CODEBOOK_SIZE=512
bash index/scripts/train.sh
