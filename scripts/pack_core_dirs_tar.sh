#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TARGET_DIRS=(docs index scripts src tokenizer)
MAX_PART_BYTES=$((10 * 1024 * 1024))

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/pack_core_dirs_tar.sh [archive_name]

Description:
  - Packs: docs/ index/ scripts/ src/ tokenizer/
  - Uses tar.gz format only.
  - Always writes output archive under repository root.
  - If archive size > 10MB, splits into 10MB parts:
      <archive>.part.000
      <archive>.part.001
      ...

Arguments:
  archive_name   Optional base name or .tar.gz name.
                 Default: grec_core_dirs_YYYYmmdd_HHMMSS.tar.gz
USAGE
}

tar_supports_flag() {
  tar --help 2>&1 | grep -Fq -- "$1"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

for dir_name in "${TARGET_DIRS[@]}"; do
  if [[ ! -d "$dir_name" ]]; then
    echo "Error: required directory not found: $dir_name" >&2
    exit 1
  fi
done

if [[ $# -eq 1 ]]; then
  archive_basename="$(basename "$1")"
else
  archive_basename="grec_core_dirs_$(date +%Y%m%d_%H%M%S).tar.gz"
fi

if [[ "$archive_basename" != *.tar.gz ]]; then
  archive_basename="${archive_basename}.tar.gz"
fi

archive_path="${PROJECT_ROOT}/${archive_basename}"

rm -f "$archive_path"

tar_create_opts=()
if tar_supports_flag "--no-xattrs"; then
  tar_create_opts+=(--no-xattrs)
fi
if tar_supports_flag "--no-mac-metadata"; then
  tar_create_opts+=(--no-mac-metadata)
fi
if tar_supports_flag "--disable-copyfile"; then
  tar_create_opts+=(--disable-copyfile)
fi

if (( ${#tar_create_opts[@]} > 0 )); then
  COPYFILE_DISABLE=1 tar "${tar_create_opts[@]}" -czf "$archive_path" "${TARGET_DIRS[@]}"
else
  COPYFILE_DISABLE=1 tar -czf "$archive_path" "${TARGET_DIRS[@]}"
fi

archive_bytes=$(wc -c < "$archive_path" | tr -d '[:space:]')

if (( archive_bytes <= MAX_PART_BYTES )); then
  echo "Created archive: $archive_path"
  echo "Archive size: ${archive_bytes} bytes (<= 10MB, no split needed)"
  exit 0
fi

if ! command -v split >/dev/null 2>&1; then
  echo "Error: archive is larger than 10MB but 'split' command is not available." >&2
  echo "Generated unsplit archive: $archive_path" >&2
  exit 1
fi

part_prefix="${archive_path}.part."
rm -f "${part_prefix}"*
split -b "$MAX_PART_BYTES" -d -a 3 "$archive_path" "$part_prefix"
rm -f "$archive_path"

part_count=$(ls -1 "${part_prefix}"* | wc -l | tr -d '[:space:]')

echo "Created split archive parts (${part_count} files):"
ls -lh "${part_prefix}"*
echo "Reassemble command: cat ${part_prefix}* > ${archive_path}"
