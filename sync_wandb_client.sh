#!/usr/bin/env bash
# local_wandb_merge.sh
# 用法：./local_wandb_merge.sh  ~/Downloads/wandb_2024-08-31-120000.tar.gz
set -e

ARCHIVE="$1"
[[ -z $ARCHIVE ]] && { echo "Usage: $0 <path_to_tar.gz>"; exit 1; }

# 本地已同步的根目录
LOCAL_ROOT="./wandb"
mkdir -p "$LOCAL_ROOT"

# 临时解压目录
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo ">>> 解压 $ARCHIVE ..."
tar -xzf "$ARCHIVE" -C "$TMP"

# 找到解压后的 wandb 根（兼容 tar 里多一层或少一层）
WANDB_SRC=$(find "$TMP" -type d -name wandb -print -quit || true)
[[ -z $WANDB_SRC ]] && WANDB_SRC="$TMP"   # 如果 tar 根就是 wandb

echo ">>> 增量合并到 $LOCAL_ROOT ..."
rsync -a --update "$WANDB_SRC/" "$LOCAL_ROOT/"

echo ">>> 完成。"