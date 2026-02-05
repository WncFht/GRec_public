#!/bin/bash
set -euo pipefail

# Train ONE ResKmeans tokenizer on multiple Amazon datasets' embeddings,
# then export per-dataset index JSON files under each dataset directory.
#
# Expected embedding outputs (from src/GRec/index/amazon_text2emb.py):
#   $DATA_ROOT/$DATASET/${DATASET}.emb-${PLM_NAME}-td.npy
#   $DATA_ROOT/$DATASET/${DATASET}.emb-${PLM_NAME}-td.ids.json   (recommended)
#
# Usage:
#   DATA_ROOT=/path/to/data PLM_NAME=qwen3-embedding-4B bash amazon_train_and_export_index.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage:
  bash amazon_train_and_export_index.sh [all|check|train|export]

Modes:
  all    (default) check -> train -> export
  check            only check required files exist
  train            only train tokenizer
  export           only export per-dataset index json (requires trained tokenizer)

Config (env vars, see amazon_index_env.sh):
  DATA_ROOT, PLM_NAME, DATASETS_STR, N_LAYERS, CODEBOOK_SIZE, DIM, NITER,
  MAX_TRAIN_POINTS, MAX_POINTS_PER_CENTROID, FAISS_GPU, DEVICE, BATCH_SIZE,
  INDEX_NAME, TOKENIZER_OUT, MODEL_PT
EOF
}

MODE="${1:-all}"
case "$MODE" in
  all|check|train|export) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "$MODE" == "all" || "$MODE" == "check" ]]; then
  bash "$SCRIPT_DIR/amazon_check_embeddings.sh"
fi

if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  bash "$SCRIPT_DIR/amazon_train_tokenizer.sh"
fi

if [[ "$MODE" == "all" || "$MODE" == "export" ]]; then
  bash "$SCRIPT_DIR/amazon_export_index_json.sh"
fi
