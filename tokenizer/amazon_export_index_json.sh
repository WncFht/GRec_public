#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/amazon_index_env.sh"

echo_config

MODEL_PT="${MODEL_PT:-$(tokenizer_model_pt)}"
if [[ ! -f "$MODEL_PT" ]]; then
  echo "Tokenizer checkpoint not found: $MODEL_PT" >&2
  echo "Hint: run ./amazon_train_tokenizer.sh first, or set MODEL_PT=/path/to/model.pt" >&2
  exit 1
fi

echo "========== Export per-dataset index JSON =========="
for d in "${DATASETS[@]}"; do
  emb="$(emb_path_for_dataset "$d")"
  ids="$(ids_path_for_dataset "$d")"
  out="$(index_json_path_for_dataset "$d")"

  if [[ ! -f "$emb" ]]; then
    echo "Missing embedding file: $emb" >&2
    exit 1
  fi

  IDS_ARG=()
  if [[ -f "$ids" ]]; then
    IDS_ARG=(--ids_path "$ids")
  fi

  echo "---- $d ----"
  python3 "$SCRIPT_DIR/build_index_json.py" \
    --model_path "$MODEL_PT" \
    --emb_path "$emb" \
    "${IDS_ARG[@]}" \
    --output_path "$out" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
done

echo "Done."

