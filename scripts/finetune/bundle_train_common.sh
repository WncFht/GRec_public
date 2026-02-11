#!/bin/bash
set -eo pipefail

normalize_model_tag() {
  local raw="$1"
  raw="$(basename "$raw")"
  raw="${raw,,}"
  raw="${raw// /-}"
  raw="${raw//\//-}"
  raw="${raw//_/-}"
  raw="$(echo "$raw" | sed -E 's/[^a-z0-9.-]+/-/g; s/-+/-/g; s/^-+//; s/-+$//')"
  printf '%s' "$raw"
}

if [[ -z "${INDEX_TAG:-}" ]]; then
  echo "Error: INDEX_TAG is required (export INDEX_TAG before calling)." >&2
  exit 1
fi

DEFAULT_GREC_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec"
: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

cd "$GREC_ROOT" || exit 1

INDEX_MATCH_TAG="${INDEX_TAG%%_sk*}"
: "${DATASET:=Instruments}"
: "${DATA_PATH:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data}"
: "${ROOT_DIR:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"
: "${INDEX_EMB_MODEL:=qwen3-embedding-4B}"
: "${INDEX_DATASETS:=Instruments}"
: "${BASE_MODEL:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/ckpt/base_model/Qwen2.5-3B-Instruct}"

if [[ -z "${SFT_MODEL_TAG:-}" ]]; then
  SFT_MODEL_TAG="$(normalize_model_tag "$BASE_MODEL")"
fi
if [[ -z "$SFT_MODEL_TAG" ]]; then
  echo "Error: failed to derive SFT_MODEL_TAG from BASE_MODEL=$BASE_MODEL" >&2
  echo "Hint: export SFT_MODEL_TAG explicitly (e.g., qwen2.5-7b-instruct)." >&2
  exit 1
fi

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
TASKS_DEFAULT="item2index,seqrec,fusionseqrec"
TASKS_FOR_TAG="${TASKS:-$TASKS_DEFAULT}"
TASKS_TAG="${TASKS_FOR_TAG// /}"
TASKS_TAG="${TASKS_TAG//,/-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR_DEFAULT="${ROOT_DIR}/ckpt/$DATASET/${SFT_MODEL_TAG}-sft__tasks-${TASKS_TAG}__idx-${INDEX_KEY}__rid-${RUN_ID}"

export DATASET DATA_PATH INDEX_FILE BASE_MODEL SFT_MODEL_TAG
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
export EVAL_BY_DATASET="${EVAL_BY_DATASET:-true}"
export EVAL_MAIN_DATASET="${EVAL_MAIN_DATASET:-$DATASET}"

echo "[bundle/train] INDEX_TAG=$INDEX_TAG"
echo "[bundle/train] DATASET=$DATASET INDEX_FILE=$INDEX_FILE"
echo "[bundle/train] SFT_MODEL_TAG=$SFT_MODEL_TAG BASE_MODEL=$BASE_MODEL"
echo "[bundle/train] INDEX_EMB_MODEL=$INDEX_EMB_MODEL INDEX_DATASETS=$INDEX_DATASETS"
echo "[bundle/train] OUTPUT_DIR=$OUTPUT_DIR"
echo "[bundle/train] EVAL_BY_DATASET=$EVAL_BY_DATASET EVAL_MAIN_DATASET=$EVAL_MAIN_DATASET"

bash "$GREC_ROOT/scripts/finetune/train_text.sh" "$@"
