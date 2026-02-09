#!/bin/bash
set -euo pipefail

DEFAULT_GREC_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec"
: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

cd "$GREC_ROOT" || exit 1

INDEX_TAG="rq4_cb128-128-128-128_sk0.0-0.0-0.0-0.003"
INDEX_MATCH_TAG="${INDEX_TAG%%_sk*}"
: "${DATASET:=Instruments}"
: "${DATA_PATH:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data}"
: "${INDEX_EMB_MODEL:=qwen3-embedding-4B}"
: "${INDEX_DATASETS:=Instruments}"

INDEX_DATASETS_TAG="${INDEX_DATASETS// /}"
INDEX_DATASETS_TAG="${INDEX_DATASETS_TAG//,/-}"

if [[ -z "${INDEX_FILE:-}" ]]; then
  pattern="$DATA_PATH/$DATASET/${DATASET}.index_emb-${INDEX_EMB_MODEL}_${INDEX_MATCH_TAG}_ds${INDEX_DATASETS_TAG}_rid*.json"
  latest_index="$(ls -1t $pattern 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_index" ]]; then
    echo "Error: cannot find index file for tag $INDEX_TAG under $DATA_PATH/$DATASET" >&2
    echo "Hint: run index generate first or set INDEX_FILE manually." >&2
    exit 1
  fi
  base_name="$(basename "$latest_index")"
  INDEX_FILE=".${base_name#${DATASET}.}"
fi

INDEX_KEY="${INDEX_FILE#.}"
INDEX_KEY="${INDEX_KEY%.json}"
OUTPUT_DIR_DEFAULT="./ckpt/$DATASET/qwen2.5-3b-sft__idx-$INDEX_KEY"

export DATASET DATA_PATH INDEX_FILE
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"

echo "[bundle/train] DATASET=$DATASET INDEX_FILE=$INDEX_FILE"
echo "[bundle/train] INDEX_EMB_MODEL=$INDEX_EMB_MODEL INDEX_DATASETS=$INDEX_DATASETS"
echo "[bundle/train] OUTPUT_DIR=$OUTPUT_DIR"

bash "$GREC_ROOT/scripts/finetune/train_text.sh" "$@"
