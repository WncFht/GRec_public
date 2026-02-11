#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INDEX_TAG="rq4_cb512-512-512-512_sk0.0-0.0-0.0-0.003"

bash "$SCRIPT_DIR/../bundle_metric_common.sh" "$@"
