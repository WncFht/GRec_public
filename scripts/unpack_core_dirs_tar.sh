#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec"

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
  3) Extract tar.gz into /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec and overwrite content.
  4) Remove source archive file(s) after successful extraction.
  5) Clean all grec_core_dirs*.tar.gz and split parts in source directory.
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
source_archives=()
source_dir=""

cleanup() {
  if [[ -n "$temp_archive" && -f "$temp_archive" ]]; then
    rm -f "$temp_archive"
  fi
}
trap cleanup EXIT

if [[ -f "$input_path" && "$input_path" == *.tar.gz ]]; then
  archive_path="$input_path"
  source_archives=("$input_path")
  source_dir="$(dirname "$input_path")"
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
  source_archives=("${part_files[@]}")
  source_dir="$(dirname "${part_files[0]}")"
  echo "Rebuilt archive from ${#part_files[@]} part files."
fi

if [[ ! -d "$DEST_ROOT" ]]; then
  echo "Error: destination root does not exist: $DEST_ROOT" >&2
  exit 1
fi

cd "$DEST_ROOT"

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

for archive_file in "${source_archives[@]}"; do
  if [[ -f "$archive_file" ]]; then
    rm -f "$archive_file"
  fi
done

if [[ -n "$source_dir" ]]; then
  shopt -s nullglob
  grec_archives=(
    "$source_dir"/grec_core_dirs_*.tar.gz
    "$source_dir"/grec_core_dirs_*.tar.gz.part.[0-9][0-9][0-9]
  )
  shopt -u nullglob

  if [[ ${#grec_archives[@]} -gt 0 ]]; then
    rm -f "${grec_archives[@]}"
    echo "Removed all grec archive file(s) in source dir: ${#grec_archives[@]}"
  fi
fi

echo "Extraction complete in $DEST_ROOT"
echo "Overwritten directories: ${TARGET_DIRS[*]}"
if (( ${#source_archives[@]} > 0 )); then
  echo "Removed source archive file(s): ${source_archives[*]}"
fi
