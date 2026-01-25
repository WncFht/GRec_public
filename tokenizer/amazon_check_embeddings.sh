#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/amazon_index_env.sh"

echo_config

echo "========== Check embeddings =========="
collect_emb_paths_or_die
echo "OK: Found ${#EMB_PATHS[@]} embedding files."

missing_ids=0
for d in "${DATASETS[@]}"; do
  ids="$(ids_path_for_dataset "$d")"
  if [[ ! -f "$ids" ]]; then
    echo "[warn] Missing ids file (will fall back to 0..N-1): $ids" >&2
    missing_ids=$((missing_ids + 1))
  fi
done
echo "Done. Missing ids files: $missing_ids"

