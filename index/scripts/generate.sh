#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

: "${ROOT_DIR:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"
: "${DATA_ROOT:=${ROOT_DIR}/data}"

: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${CKPT_PATH:=}"
if [[ -z "$CKPT_PATH" ]]; then
  echo "Error: CKPT_PATH is empty. Please export CKPT_PATH to a trained checkpoint path."
  exit 1
fi

: "${USE_MULTI_DATASETS:=true}"
: "${DATASET:=Instruments}"
: "${DATA_PATH:=${DATA_ROOT}/${DATASET}/${DATASET}.emb-${MODEL_NAME}-td.npy}"
: "${DATASETS:=Arts Automotive Cell Games Pet Sports Tools Toys Instruments}"
: "${OUTPUT_SUFFIX:=.index_${MODEL_NAME}.json}"
: "${DEVICE:=cuda:0}"
: "${BATCH_SIZE:=64}"

read -r -a DATASET_LIST <<< "$DATASETS"
DATA_PATHS=(${DATA_PATHS:-})

gen_one() {
  local dataset="$1"
  local data_path="$2"
  local output_dir="${DATA_ROOT}/${dataset}/"
  local output_file="${dataset}${OUTPUT_SUFFIX}"

  python3 -m index.generate_indices \
    --dataset "$dataset" \
    --ckpt_path "$CKPT_PATH" \
    --data_path "$data_path" \
    --output_dir "$output_dir" \
    --output_file "$output_file" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
}

if [ "${USE_MULTI_DATASETS,,}" = "true" ]; then
  if [ ${#DATASET_LIST[@]} -eq 0 ]; then
    echo "Error: USE_MULTI_DATASETS=true but DATASETS is empty."
    exit 1
  fi

  if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    for dataset_name in "${DATASET_LIST[@]}"; do
      DATA_PATHS+=("${DATA_ROOT}/${dataset_name}/${dataset_name}.emb-${MODEL_NAME}-td.npy")
    done
  fi

  if [ ${#DATA_PATHS[@]} -ne ${#DATASET_LIST[@]} ]; then
    echo "Error: DATASETS and DATA_PATHS must have the same length."
    exit 1
  fi

  python3 -m index.generate_indices \
    --datasets "${DATASET_LIST[@]}" \
    --ckpt_path "$CKPT_PATH" \
    --data_paths "${DATA_PATHS[@]}" \
    --output_dir "$DATA_ROOT" \
    --output_suffix "$OUTPUT_SUFFIX" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
else
  gen_one "$DATASET" "$DATA_PATH"
fi
