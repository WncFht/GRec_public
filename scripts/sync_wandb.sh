#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(realpath ./wandb)        # 绝对路径
SNAP_DIR=$(realpath ./wandb_snap)
CHUNK_SIZE=10485760                 # 10 MB
TMP_DIR="$SNAP_DIR/tmp_$(date +%s)"

mkdir -p "$SNAP_DIR"
rm -rf "$TMP_DIR"                   # 清理上一次残留
mkdir -p "$TMP_DIR"

LAST_FILE="$SNAP_DIR/.last_sync"

# --------------------------------------------------
# 1. 收集需要同步的 offline-run-*
collect_runs() {
    if [[ ! -f "$LAST_FILE" ]]; then
        find "$ROOT_DIR" -maxdepth 1 -type d -name 'offline-run-*'
    else
        find "$ROOT_DIR" -maxdepth 1 -type d -name 'offline-run-*' -newer "$LAST_FILE"
    fi
}

mapfile -t RUNS < <(collect_runs)
[[ ${#RUNS[@]} -eq 0 ]] && { echo "Nothing to package."; exit 0; }

echo ">>> 本次同步的目录："
printf '%s\n' "${RUNS[@]}"

# --------------------------------------------------
# 2. 复制到 TMP_DIR
for run in "${RUNS[@]}"; do
    base=$(basename "$run")
    cp -a "$run" "$TMP_DIR/"
    [[ -f "$run.wandb" ]] && cp -a "$run.wandb" "$TMP_DIR/"
done

# 3. 冻结时间戳，避免 tar 告警
find "$TMP_DIR" -exec touch {} +

# 4. 分块压缩
ARCHIVE_PREFIX="$SNAP_DIR/wandb_$(date +%F-%H%M%S)"
echo ">>> 开始分块压缩，每块 ≤ $(( CHUNK_SIZE / 1024 / 1024 )) MB ..."
tar -C "$TMP_DIR" -czf - . 2>/dev/null | \
    split -b "$CHUNK_SIZE" -d -a 3 - "$ARCHIVE_PREFIX.part"

# --------------------------------------------------
# 4. 更新时间戳并清理
touch "$LAST_FILE"
rm -rf "$TMP_DIR"

echo ">>> 完成！分块文件保存在 $SNAP_DIR"
ls -lh "$ARCHIVE_PREFIX".part*