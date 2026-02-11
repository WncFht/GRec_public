#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INDEX_TAG="rq4_cb32-32-32-32_sk0.0-0.0-0.0-0.003"

bash "$SCRIPT_DIR/../bundle_metric_common.sh" "$@"
