#!/usr/bin/env bash
set -euo pipefail

# 打包 wandb/offline-run-*，如果 run 后续有新文件或内容变更，会再次打包。
# 默认会同时处理:
#   1) GRec/wandb
#   2) ../GenRec/wandb
# 环境变量可覆盖:
#   GREC_ROOT_DIR=/path/to/GRec/wandb
#   GENREC_ROOT_DIR=/path/to/GenRec/wandb
#   SNAP_DIR=/path/to/GRec/wandb_snap
#   CHUNK_SIZE=10485760
# 兼容旧变量:
#   ROOT_DIR=/path/to/GRec/wandb

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SNAP_DIR=${SNAP_DIR:-"$REPO_ROOT/wandb_snap"}
GREC_ROOT_DIR=${GREC_ROOT_DIR:-${ROOT_DIR:-"$REPO_ROOT/wandb"}}
GENREC_ROOT_DIR=${GENREC_ROOT_DIR:-"$REPO_ROOT/../GenRec/wandb"}
CHUNK_SIZE=${CHUNK_SIZE:-10485760}  # 10MB 默认分块

mkdir -p "$SNAP_DIR"

TMP_DIRS=()
PACKAGED_ANY=0

cleanup() {
  local tmp_dir
  for tmp_dir in "${TMP_DIRS[@]:-}"; do
    [[ -n "$tmp_dir" ]] && rm -rf "$tmp_dir"
  done
}
trap cleanup EXIT

run_changed() {
  local run="$1"
  local last_file="$2"

  # 首次运行，全量
  [[ ! -f "$last_file" ]] && return 0

  # 目录本身是新建的
  [[ "$run" -nt "$last_file" ]] && return 0

  # 目录下有文件比上次新
  if find "$run" -type f -newer "$last_file" -print -quit | read -r _; then
    return 0
  fi

  # 对应的 run.wandb 有更新
  [[ -f "$run.wandb" && "$run.wandb" -nt "$last_file" ]] && return 0

  return 1
}

collect_runs() {
  local root_dir="$1"
  local last_file="$2"
  while IFS= read -r dir; do
    run_changed "$dir" "$last_file" && printf '%s\n' "$dir"
  done < <(find "$root_dir" -maxdepth 1 -type d -name 'offline-run-*' | LC_ALL=C sort)
}

package_source() {
  local source_name="$1"
  local root_dir="$2"
  local last_file="${SNAP_DIR}/.last_sync_${source_name}"
  local archive_prefix
  local tmp_dir
  local run
  local file
  local -a runs=()
  local -a old_parts=()

  if [[ -z "$root_dir" || ! -d "$root_dir" ]]; then
    echo ">>> [${source_name}] Skip: source dir not found: $root_dir"
    return 0
  fi

  mapfile -t runs < <(collect_runs "$root_dir" "$last_file")
  if [[ ${#runs[@]} -eq 0 ]]; then
    echo ">>> [${source_name}] Nothing to package (no new/updated offline-run-* since last sync)."
    return 0
  fi

  echo ">>> [${source_name}] 本次同步的目录："
  printf '%s\n' "${runs[@]}"

  tmp_dir="${SNAP_DIR}/tmp_${source_name}_$$_$(date +%s)"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"
  TMP_DIRS+=("$tmp_dir")

  # 复制需要的目录
  for run in "${runs[@]}"; do
    cp -a "$run" "$tmp_dir/"
    [[ -f "$run.wandb" ]] && cp -a "$run.wandb" "$tmp_dir/"
  done

  # 冻结时间戳，避免 tar 告警
  find "$tmp_dir" -exec touch {} +

  # 分块压缩
  archive_prefix="$SNAP_DIR/wandb_${source_name}_$(date +%F-%H%M%S)"
  # 避免同名前缀遗留旧分块（例如同秒重复执行）。
  rm -f "$archive_prefix".part*
  echo ">>> [${source_name}] 开始分块压缩，每块 ≤ $(( CHUNK_SIZE / 1024 / 1024 )) MB ..."
  tar -C "$tmp_dir" -czf - . 2>/dev/null | \
      split -b "$CHUNK_SIZE" -d -a 3 - "$archive_prefix.part"

  # 仅保留最新一套分块，清理当前 source 的历史压缩包。
  while IFS= read -r -d '' file; do
    [[ "$file" == "$archive_prefix".part* ]] && continue
    old_parts+=("$file")
  done < <(find "$SNAP_DIR" -maxdepth 1 -type f -name "wandb_${source_name}_*.part*" -print0)
  if [[ ${#old_parts[@]} -gt 0 ]]; then
    rm -f "${old_parts[@]}"
    echo ">>> [${source_name}] 已清理历史分块：${#old_parts[@]} 个"
  fi

  # 更新时间戳
  touch "$last_file"
  PACKAGED_ANY=1

  echo ">>> [${source_name}] 完成！分块文件保存在 $SNAP_DIR"
  ls -lh "$archive_prefix".part*
}

declare -a SOURCES=(
  "grec:$GREC_ROOT_DIR"
  "genrec:$GENREC_ROOT_DIR"
)

for source in "${SOURCES[@]}"; do
  source_name="${source%%:*}"
  source_dir="${source#*:}"
  package_source "$source_name" "$source_dir"
done

if [[ "$PACKAGED_ANY" -eq 0 ]]; then
  echo "Nothing packaged from all sources."
fi
