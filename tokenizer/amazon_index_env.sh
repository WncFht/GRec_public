#!/bin/bash
set -euo pipefail

# Common env & helpers for Amazon embedding -> tokenizer -> index export pipeline.
# This file is meant to be sourced by other scripts in this directory.

export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${DATA_ROOT:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data}"
: "${PLM_NAME:=qwen}"

# Datasets should match `index/scripts/text2emb.sh` by default.
: "${DATASETS_STR:=Arts Automotive Cell Games Instruments Pet Sports Tools Toys}"
read -r -a DATASETS <<< "$DATASETS_STR"

# Tokenizer config
: "${N_LAYERS:=3}"
: "${CODEBOOK_SIZE:=8192}"
# DIM is only used as an upper bound for slicing; train script will auto-detect and warn/override if mismatch.
: "${DIM:=2560}"
: "${NITER:=20}"
: "${MAX_TRAIN_POINTS:=0}"
: "${MAX_POINTS_PER_CENTROID:=256}"
: "${FAISS_GPU:=1}"

# Export config
: "${DEVICE:=cuda}"
: "${BATCH_SIZE:=10000}"

# Naming: index_file passed to training is usually ".index_${INDEX_NAME}.json"
: "${INDEX_NAME:=$PLM_NAME}"

# Where to save the shared tokenizer checkpoint
: "${TOKENIZER_OUT:=$DATA_ROOT/_shared_tokenizer/reskmeans_${INDEX_NAME}_L${N_LAYERS}_C${CODEBOOK_SIZE}}"

echo_config() {
  echo "DATA_ROOT:      $DATA_ROOT"
  echo "PLM_NAME:       $PLM_NAME"
  echo "INDEX_NAME:     $INDEX_NAME"
  echo "TOKENIZER_OUT:  $TOKENIZER_OUT"
  echo "N_LAYERS:       $N_LAYERS"
  echo "CODEBOOK_SIZE:  $CODEBOOK_SIZE"
  echo "DIM:            $DIM"
  echo "NITER:          $NITER"
  echo "MAX_TRAIN_PTS:  $MAX_TRAIN_POINTS"
  echo "MPSC:           $MAX_POINTS_PER_CENTROID"
  echo "FAISS_GPU:      $FAISS_GPU"
  echo "DEVICE:         $DEVICE"
  echo "BATCH_SIZE:     $BATCH_SIZE"
  echo "DATASETS:       ${DATASETS[*]}"
}

emb_path_for_dataset() {
  local d="$1"
  echo "$DATA_ROOT/$d/${d}.emb-${PLM_NAME}-td.npy"
}

ids_path_for_dataset() {
  local d="$1"
  echo "$DATA_ROOT/$d/${d}.emb-${PLM_NAME}-td.ids.json"
}

index_json_path_for_dataset() {
  local d="$1"
  echo "$DATA_ROOT/$d/${d}.index_${INDEX_NAME}.json"
}

tokenizer_model_pt() {
  echo "$TOKENIZER_OUT/model.pt"
}

collect_emb_paths_or_die() {
  EMB_PATHS=()
  for d in "${DATASETS[@]}"; do
    local p
    p="$(emb_path_for_dataset "$d")"
    if [[ ! -f "$p" ]]; then
      echo "Missing embedding file: $p" >&2
      return 1
    fi
    EMB_PATHS+=("$p")
  done
}
