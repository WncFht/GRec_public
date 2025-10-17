#!/usr/bin/env bash
# sync_wandb_client.sh
# 用法：
#   普通包   : ./sync_wandb_client.sh wandb_2024-08-31.tar.gz
#   分卷合并: ./sync_wandb_client.sh "wandb_2025-09-14-034028.part*"
set -euo pipefail

ARCHIVE_SPEC="$1"
[[ -z $ARCHIVE_SPEC ]] && { echo "Usage: $0 <path_to_tar.gz|path_to_parts*>"; exit 1; }

LOCAL_ROOT="./wandb"
mkdir -p "$LOCAL_ROOT"

# ---------- 1. 分卷检测与合并 ----------
MERGED_TAR=""
if [[ "$ARCHIVE_SPEC" == *"part"* ]]; then
    # 展开通配符并按数字序排序
    PARTS=($(ls -1v $ARCHIVE_SPEC))
    [[ ${#PARTS[@]} -eq 0 ]] && { echo "未找到任何分卷文件：$ARCHIVE_SPEC"; exit 2; }

    # 取公共前缀，防止混进不同套
    PREFIX=$(basename "${PARTS[0]}" | sed 's/\.part[0-9]*$//')
    for f in "${PARTS[@]}"; do
        [[ $(basename "$f") =~ ^$PREFIX\.part ]] || \
            { echo "分卷文件不属于同一套: $f"; exit 3; }
    done

    MERGED_TAR=$(mktemp -t "${PREFIX}.merged.XXXXXX.tar.gz")
    trap "rm -f $MERGED_TAR" EXIT

    echo ">>> 合并分卷 -> $MERGED_TAR"
    cat "${PARTS[@]}" > "$MERGED_TAR"
    ARCHIVE="$MERGED_TAR"
else
    ARCHIVE="$ARCHIVE_SPEC"
fi

# ---------- 2. 解压 ----------
TMP=$(mktemp -d)
trap "rm -rf $TMP ${MERGED_TAR:-}" EXIT

echo ">>> 解压 $ARCHIVE ..."
if file "$ARCHIVE" | grep -q gzip; then
    tar -xzf "$ARCHIVE" -C "$TMP"
else
    tar -xf "$ARCHIVE" -C "$TMP"
fi

# ---------- 3. 定位 wandb 目录 ----------
WANDB_SRC=$(find "$TMP" -type d -name wandb -print -quit || true)
[[ -z $WANDB_SRC ]] && WANDB_SRC="$TMP"

# ---------- 4. 增量同步 ----------
echo ">>> 增量合并到 $LOCAL_ROOT ..."
rsync -a --update  "$WANDB_SRC/" "$LOCAL_ROOT/"

echo ">>> 完成。"