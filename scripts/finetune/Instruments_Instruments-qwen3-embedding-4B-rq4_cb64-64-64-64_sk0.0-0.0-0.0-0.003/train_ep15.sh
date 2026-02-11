#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export INDEX_TAG="rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003"
export EPOCHS="${EPOCHS:-15}"
export RUN_ID="${RUN_ID:-ep15_$(date +%Y%m%d_%H%M%S)}"

bash "$SCRIPT_DIR/../bundle_train_common.sh" "$@"
