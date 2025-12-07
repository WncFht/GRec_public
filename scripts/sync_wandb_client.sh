#!/usr/bin/env bash
set -euo pipefail

# Merge wandb offline archives (single .tar.gz or split .part* set) into
# ~/Downloads/wandb by untarring to a temp dir and rsync'ing with --update.
# Examples:
#   ./merge_wandb_dirs.sh wandb_2025-09-14-034028.part*
#   ./merge_wandb_dirs.sh ~/Downloads/wandb_2025-09-14-034028.tar.gz
#   ./merge_wandb_dirs.sh --dest /path/to/wandb wandb_*.tar.gz

BASE_DIR="${HOME}/Downloads"
DEST_DIR_DEFAULT="${BASE_DIR}/wandb"

usage() {
  cat <<'EOF'
Usage: merge_wandb_dirs.sh [--dest DIR] <archive.tar.gz|parts-pattern> [...]
Notes:
  - If a pattern is relative and not found in cwd, the script also searches ~/Downloads.
  - Split parts must belong to the same set (PREFIX.part000, PREFIX.part001, ...).
EOF
  exit 1
}

DEST_DIR="$DEST_DIR_DEFAULT"
SPECS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      [[ $# -lt 2 ]] && { echo "Missing value for --dest" >&2; usage; }
      DEST_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      SPECS+=("$1")
      shift
      ;;
  esac
done

[[ ${#SPECS[@]} -eq 0 ]] && usage

mkdir -p "$DEST_DIR"

declare -a TMP_DIRS=()
declare -a TMP_FILES=()
cleanup() {
  for d in "${TMP_DIRS[@]:-}"; do
    [[ -d "$d" ]] && rm -rf "$d"
  done
  for f in "${TMP_FILES[@]:-}"; do
    [[ -f "$f" ]] && rm -f "$f"
  done
}
trap cleanup EXIT

expand_matches() {
  local spec="$1"
  while IFS= read -r match; do
    printf '%s\n' "$match"
  done < <(compgen -G "$spec")

  # 如果当前目录没找到，尝试 BASE_DIR
  if ! compgen -G "$spec" >/dev/null; then
    while IFS= read -r match; do
      printf '%s\n' "$match"
    done < <(compgen -G "$BASE_DIR/$spec")
  fi
}

merge_parts() {
  local -a parts=("$@")
  local -a sorted_parts=()
  while IFS= read -r part; do
    sorted_parts+=("$part")
  done < <(printf '%s\n' "${parts[@]}" | LC_ALL=C sort)
  local prefix
  prefix="$(basename "${sorted_parts[0]}")"
  prefix="${prefix%%.part*}"

  for f in "${sorted_parts[@]}"; do
    [[ $(basename "$f") =~ ^${prefix}\.part[0-9]+$ ]] || {
      echo "分卷文件不属于同一套: $f" >&2
      exit 3
    }
  done

  local merged_tmp
  merged_tmp=$(mktemp) || {
    echo "mktemp 创建临时文件失败" >&2
    exit 4
  }
  local merged="${merged_tmp}.tar.gz"
  mv "$merged_tmp" "$merged"
  TMP_FILES+=("$merged")

  echo ">>> 合并分卷 -> $merged" >&2
  cat "${sorted_parts[@]}" > "$merged"
  printf '%s\n' "$merged"
}

collect_part_set() {
  local first_part="$1"
  local dir prefix
  dir="$(cd "$(dirname "$first_part")" && pwd)"
  prefix="$(basename "$first_part")"
  prefix="${prefix%%.part*}"
  find "$dir" -maxdepth 1 -type f -name "${prefix}.part*" -print | LC_ALL=C sort
}

extract_and_sync() {
  local archive="$1"
  local tmp_dir
  tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/wandb_extract.XXXXXX")
  TMP_DIRS+=("$tmp_dir")

  echo ">>> 解压 $archive ..."
  if file --mime-type -b "$archive" | grep -q gzip; then
    tar -xzf "$archive" -C "$tmp_dir"
  else
    tar -xf "$archive" -C "$tmp_dir"
  fi

  local wandb_src
  wandb_src=$(find "$tmp_dir" -type d -name wandb -print -quit || true)
  [[ -z $wandb_src ]] && wandb_src="$tmp_dir"

  echo ">>> 增量合并到 $DEST_DIR ..."
  rsync -a --update "$wandb_src"/ "$DEST_DIR"/
}

process_spec() {
  local spec="$1"
  local matches=()
  while IFS= read -r m; do
    matches+=("$m")
  done < <(expand_matches "$spec")
  [[ ${#matches[@]} -eq 0 ]] && { echo "未找到文件: $spec (尝试过 cwd 和 $BASE_DIR)" >&2; return 1; }

  # 如果是分卷，则将全部匹配作为一套处理
  local has_part=false
  for f in "${matches[@]}"; do
    if [[ $(basename "$f") == *.part* ]]; then
      has_part=true
      break
    fi
  done

  if $has_part; then
    # 自动补全同前缀的所有分卷（无需输入 *）
    local part_set=()
    while IFS= read -r p; do
      part_set+=("$p")
    done < <(collect_part_set "${matches[0]}")

    [[ ${#part_set[@]} -eq 0 ]] && { echo "未找到分卷文件: $spec" >&2; return 1; }
    local merged
    merged=$(merge_parts "${part_set[@]}")
    extract_and_sync "$merged"
  else
    # 多个常规归档时逐个处理
    for archive in "${matches[@]}"; do
      extract_and_sync "$archive"
    done
  fi
}

for spec in "${SPECS[@]}"; do
  process_spec "$spec"
done

echo ">>> 完成。已合并到 $DEST_DIR"
