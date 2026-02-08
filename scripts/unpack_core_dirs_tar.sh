#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGET_DIRS=(docs index scripts src tokenizer)

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/unpack_core_dirs_tar.sh <archive_or_part>

Examples:
  bash scripts/unpack_core_dirs_tar.sh grec_core_dirs.tar.gz
  bash scripts/unpack_core_dirs_tar.sh grec_core_dirs.tar.gz.part.000

Behavior:
  1) Rebuild archive if split parts are provided.
  2) Remove existing docs/index/scripts/src/tokenizer directories.
  3) Extract tar.gz into repository root and overwrite content.
USAGE
}

tar_supports_flag() {
  tar --help 2>&1 | grep -Fq -- "$1"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

input_arg="$1"
current_dir="$(pwd)"

if [[ "$input_arg" = /* ]]; then
  input_path="$input_arg"
else
  input_path="${current_dir}/${input_arg}"
fi

temp_archive=""
archive_path=""

cleanup() {
  if [[ -n "$temp_archive" && -f "$temp_archive" ]]; then
    rm -f "$temp_archive"
  fi
}
trap cleanup EXIT

if [[ -f "$input_path" && "$input_path" == *.tar.gz ]]; then
  archive_path="$input_path"
else
  if [[ -f "$input_path" && "$input_path" =~ \.part\.[0-9]{3}$ ]]; then
    part_prefix="${input_path%[0-9][0-9][0-9]}"
  else
    part_prefix="$input_path"
  fi

  shopt -s nullglob
  part_files=("${part_prefix}"[0-9][0-9][0-9])
  shopt -u nullglob

  if [[ ${#part_files[@]} -eq 0 ]]; then
    echo "Error: cannot find archive or split parts from input: $input_arg" >&2
    exit 1
  fi

  temp_archive="/tmp/grec_unpack_$(date +%s)_$$.tar.gz"
  cat "${part_files[@]}" > "$temp_archive"
  archive_path="$temp_archive"
  echo "Rebuilt archive from ${#part_files[@]} part files."
fi

cd "$PROJECT_ROOT"

for dir_name in "${TARGET_DIRS[@]}"; do
  if [[ -e "$dir_name" ]]; then
    rm -rf "$dir_name"
  fi
done

tar_extract_opts=()
if tar_supports_flag "--warning"; then
  tar_extract_opts+=(--warning=no-unknown-keyword)
fi

if (( ${#tar_extract_opts[@]} > 0 )); then
  tar "${tar_extract_opts[@]}" -xzf "$archive_path"
else
  tar -xzf "$archive_path"
fi

echo "Extraction complete in $PROJECT_ROOT"
echo "Overwritten directories: ${TARGET_DIRS[*]}"
