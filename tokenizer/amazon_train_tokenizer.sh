#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/amazon_index_env.sh"

echo_config

echo "========== Train tokenizer =========="
collect_emb_paths_or_die

FAISS_GPU_FLAG=()
if [[ "$FAISS_GPU" == "1" ]]; then
  FAISS_GPU_FLAG=(--faiss_gpu)
fi

echo $FAISS_GPU_FLAG

python3 "$SCRIPT_DIR/train_res_kmeans.py" \
  --data_paths "${EMB_PATHS[@]}" \
  --model_path "$TOKENIZER_OUT" \
  --n_layers "$N_LAYERS" \
  --codebook_size "$CODEBOOK_SIZE" \
  --dim "$DIM" \
  --niter "$NITER" \
  --max_train_points "$MAX_TRAIN_POINTS" \
  --max_points_per_centroid "$MAX_POINTS_PER_CENTROID" \
  "${FAISS_GPU_FLAG[@]}"

MODEL_PT="$(tokenizer_model_pt)"
if [[ ! -f "$MODEL_PT" ]]; then
  echo "Tokenizer checkpoint not found: $MODEL_PT" >&2
  exit 1
fi

echo "OK: Tokenizer saved to $MODEL_PT"

