#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE_HOME_ROOT_DEFAULT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian"

LOCAL_ROOT="${LOCAL_ROOT:-$LOCAL_ROOT_DEFAULT}"
REMOTE_HOME_ROOT="${REMOTE_HOME_ROOT:-$REMOTE_HOME_ROOT_DEFAULT}"
MAX_FILE_MB="${MAX_FILE_MB:-10}"
EXCLUDE_BASENAMES_CSV="${EXCLUDE_BASENAMES_CSV:-final_result.json,rollout.json}"
CHUNK_SIZE="${CHUNK_SIZE:-10485760}"  # 10MB

REMOTE_GREC_SEQREC_DIR="${REMOTE_GREC_SEQREC_DIR:-$REMOTE_HOME_ROOT/GRec/results/test/seqrec-constrained}"
REMOTE_GENREC_RESULTS_DIR="${REMOTE_GENREC_RESULTS_DIR:-$REMOTE_HOME_ROOT/GenRec/results}"

LOCAL_GREC_SEQREC_DIR="${LOCAL_GREC_SEQREC_DIR:-$LOCAL_ROOT/GRec_public/results/test/seqrec-constrained}"
LOCAL_GENREC_RESULTS_DIR="${LOCAL_GENREC_RESULTS_DIR:-$LOCAL_ROOT/GenRec/results}"

usage() {
  cat <<'USAGE'
Usage:
  # Step 1 (run on training machine): pack two results dirs into one tar.gz
  bash scripts/sync_results_from_remote.sh pack [archive_name]

  # Step 2 (after manual download, run on local machine): unpack and overwrite local results
  bash scripts/sync_results_from_remote.sh unpack [archive_or_part]

Description:
  - No SSH sync. This script uses manual transfer workflow only.
  - pack:
      1) Packs remote GRec/results/test/seqrec-constrained
      2) Packs remote GenRec/results
      3) Ignores files larger than MAX_FILE_MB (default 10MB)
      4) Ignores basenames in EXCLUDE_BASENAMES_CSV (default: final_result.json,rollout.json)
      5) Builds tar.gz; if final archive > CHUNK_SIZE (default 10MB), splits into .part.000/.001/...
      6) Default archive is results_sync_latest.tar.gz and will overwrite old default archives
  - unpack:
      1) If input is omitted (or not a .tar.gz path), auto-picks latest result*.tar.gz in current dir
      2) Extracts archive (or rebuilds from .part.000 + parts)
      3) Overwrites local:
         <LOCAL_ROOT>/GRec_public/results/test/seqrec-constrained
         <LOCAL_ROOT>/GenRec/results
      4) Removes downloaded results_*.tar.gz (and split parts) in the source directory after success

Environment overrides:
  REMOTE_HOME_ROOT
  REMOTE_GREC_SEQREC_DIR
  REMOTE_GENREC_RESULTS_DIR
  MAX_FILE_MB
  EXCLUDE_BASENAMES_CSV
  CHUNK_SIZE
  LOCAL_ROOT
  LOCAL_GREC_SEQREC_DIR
  LOCAL_GENREC_RESULTS_DIR
USAGE
}

tar_supports_flag() {
  tar --help 2>&1 | grep -Fq -- "$1"
}

find_latest_results_archive() {
  local search_dir="$1"
  local -a candidate_archives=()
  local latest_archive

  shopt -s nullglob
  candidate_archives=("$search_dir"/result*.tar.gz)
  shopt -u nullglob

  if [[ ${#candidate_archives[@]} -eq 0 ]]; then
    return 1
  fi

  latest_archive="$(ls -1t "${candidate_archives[@]}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_archive" ]]; then
    return 1
  fi

  printf '%s\n' "$latest_archive"
}

pack_results() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  if [[ $# -gt 1 ]]; then
    usage
    exit 1
  fi

  if ! [[ "$MAX_FILE_MB" =~ ^[0-9]+$ ]] || (( MAX_FILE_MB <= 0 )); then
    echo "Error: MAX_FILE_MB must be a positive integer, got: $MAX_FILE_MB" >&2
    exit 1
  fi
  if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || (( CHUNK_SIZE <= 0 )); then
    echo "Error: CHUNK_SIZE must be a positive integer (bytes), got: $CHUNK_SIZE" >&2
    exit 1
  fi

  if [[ ! -d "$REMOTE_HOME_ROOT" ]]; then
    echo "Error: REMOTE_HOME_ROOT not found: $REMOTE_HOME_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$REMOTE_GREC_SEQREC_DIR" ]]; then
    echo "Error: source dir not found: $REMOTE_GREC_SEQREC_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$REMOTE_GENREC_RESULTS_DIR" ]]; then
    echo "Error: source dir not found: $REMOTE_GENREC_RESULTS_DIR" >&2
    exit 1
  fi

  local archive_arg archive_basename archive_path
  archive_arg="${1:-results_sync_latest.tar.gz}"

  if [[ "$archive_arg" = /* ]]; then
    archive_path="$archive_arg"
  else
    archive_path="$(pwd)/$archive_arg"
  fi

  if [[ "$archive_path" != *.tar.gz ]]; then
    archive_path="${archive_path}.tar.gz"
  fi

  local archive_dir
  archive_dir="$(dirname "$archive_path")"

  if [[ $# -eq 0 ]]; then
    local -a old_default_archives=()
    shopt -s nullglob
    old_default_archives=(
      "$archive_dir"/results_sync_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].tar.gz
      "$archive_dir"/results_sync_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].tar.gz.part.[0-9][0-9][0-9]
      "$archive_dir"/results_sync_latest.tar.gz
      "$archive_dir"/results_sync_latest.tar.gz.part.[0-9][0-9][0-9]
    )
    shopt -u nullglob

    if [[ ${#old_default_archives[@]} -gt 0 ]]; then
      rm -f "${old_default_archives[@]}"
      echo "Removed old default archive file(s): ${#old_default_archives[@]}"
    fi
  fi

  local grec_rel_path genrec_rel_path
  grec_rel_path="${REMOTE_GREC_SEQREC_DIR#"$REMOTE_HOME_ROOT/"}"
  genrec_rel_path="${REMOTE_GENREC_RESULTS_DIR#"$REMOTE_HOME_ROOT/"}"

  if [[ "$grec_rel_path" == "$REMOTE_GREC_SEQREC_DIR" || "$genrec_rel_path" == "$REMOTE_GENREC_RESULTS_DIR" ]]; then
    echo "Error: source dirs must be under REMOTE_HOME_ROOT=$REMOTE_HOME_ROOT" >&2
    exit 1
  fi

  mkdir -p "$archive_dir"
  rm -f "$archive_path" "${archive_path}.part."[0-9][0-9][0-9]

  local -a exclude_basenames=()
  if [[ -n "$EXCLUDE_BASENAMES_CSV" ]]; then
    local -a raw_exclude_names
    local raw_name trimmed_name
    IFS=',' read -r -a raw_exclude_names <<< "$EXCLUDE_BASENAMES_CSV"
    for raw_name in "${raw_exclude_names[@]}"; do
      trimmed_name="${raw_name#"${raw_name%%[![:space:]]*}"}"
      trimmed_name="${trimmed_name%"${trimmed_name##*[![:space:]]}"}"
      if [[ -n "$trimmed_name" ]]; then
        exclude_basenames+=("$trimmed_name")
      fi
    done
  fi

  local max_file_bytes
  max_file_bytes=$((MAX_FILE_MB * 1024 * 1024))

  local tmp_file_list
  tmp_file_list="/tmp/results_sync_pack_files_$(date +%s)_$$.txt"
  rm -f "$tmp_file_list"

  local all_files_list
  all_files_list="/tmp/results_sync_pack_all_files_$(date +%s)_$$.txt"
  rm -f "$all_files_list"
  echo "Scanning source files..."
  find "$REMOTE_GREC_SEQREC_DIR" "$REMOTE_GENREC_RESULTS_DIR" -type f | LC_ALL=C sort -u > "$all_files_list"

  local total_files kept_files skipped_large_files skipped_name_files
  total_files=$(wc -l < "$all_files_list" | tr -d '[:space:]')
  kept_files=0
  skipped_large_files=0
  skipped_name_files=0

  local abs_file rel_file base_name excluded_name file_bytes is_excluded
  while IFS= read -r abs_file; do
    if [[ -z "$abs_file" ]]; then
      continue
    fi

    base_name="$(basename "$abs_file")"
    is_excluded=0
    for excluded_name in "${exclude_basenames[@]}"; do
      if [[ "$base_name" == "$excluded_name" ]]; then
        is_excluded=1
        break
      fi
    done

    if (( is_excluded == 1 )); then
      skipped_name_files=$((skipped_name_files + 1))
      continue
    fi

    file_bytes=$(wc -c < "$abs_file" | tr -d '[:space:]')
    if (( file_bytes > max_file_bytes )); then
      skipped_large_files=$((skipped_large_files + 1))
      continue
    fi

    rel_file="${abs_file#"$REMOTE_HOME_ROOT/"}"
    if [[ "$rel_file" == "$abs_file" ]]; then
      rm -f "$tmp_file_list" "$all_files_list"
      echo "Error: file is not under REMOTE_HOME_ROOT: $abs_file" >&2
      exit 1
    fi

    printf '%s\n' "$rel_file" >> "$tmp_file_list"
    kept_files=$((kept_files + 1))
  done < "$all_files_list"

  if (( kept_files == 0 )); then
    rm -f "$tmp_file_list"
    rm -f "$all_files_list"
    echo "Error: no files matched pack filters." >&2
    echo "Check MAX_FILE_MB and EXCLUDE_BASENAMES_CSV." >&2
    exit 1
  fi

  local -a tar_create_opts
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

  echo "Packing:"
  echo "  - $REMOTE_GREC_SEQREC_DIR"
  echo "  - $REMOTE_GENREC_RESULTS_DIR"
  echo "Filter: file_size <= ${MAX_FILE_MB}MB"
  if (( ${#exclude_basenames[@]} > 0 )); then
    echo "Filter: exclude basenames: ${exclude_basenames[*]}"
  else
    echo "Filter: exclude basenames: (none)"
  fi
  echo "Files: total=$total_files kept=$kept_files skipped_large=$skipped_large_files skipped_by_name=$skipped_name_files"
  echo "Archive: $archive_path"

  if (( ${#tar_create_opts[@]} > 0 )); then
    COPYFILE_DISABLE=1 tar "${tar_create_opts[@]}" -czf "$archive_path" -C "$REMOTE_HOME_ROOT" -T "$tmp_file_list"
  else
    COPYFILE_DISABLE=1 tar -czf "$archive_path" -C "$REMOTE_HOME_ROOT" -T "$tmp_file_list"
  fi

  rm -f "$tmp_file_list" "$all_files_list"

  local archive_bytes
  archive_bytes=$(wc -c < "$archive_path" | tr -d '[:space:]')
  archive_basename="$(basename "$archive_path")"
  if (( archive_bytes <= CHUNK_SIZE )); then
    echo "Created archive: $archive_path"
    echo "Archive basename: $archive_basename"
    echo "Archive size: ${archive_bytes} bytes"
    echo "OUTPUT_ARCHIVE=$archive_path"
    echo "Next: manually download this archive to your local machine, then run unpack."
    return
  fi

  if ! command -v split >/dev/null 2>&1; then
    echo "Error: archive size ${archive_bytes} bytes exceeds CHUNK_SIZE=${CHUNK_SIZE}, but 'split' is not available." >&2
    echo "Generated unsplit archive: $archive_path" >&2
    exit 1
  fi

  local part_prefix
  part_prefix="${archive_path}.part."
  rm -f "${part_prefix}"*
  split -b "$CHUNK_SIZE" -d -a 3 "$archive_path" "$part_prefix"
  rm -f "$archive_path"

  local part_count
  part_count=$(ls -1 "${part_prefix}"* | wc -l | tr -d '[:space:]')
  echo "Archive size ${archive_bytes} bytes > CHUNK_SIZE=${CHUNK_SIZE}, split into ${part_count} parts:"
  ls -lh "${part_prefix}"*
  echo "OUTPUT_PART_PREFIX=${part_prefix}"
  echo "Next: manually download all .part.* files, then run unpack with .part.000 (or prefix path)."
}

unpack_results() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  if [[ $# -gt 1 ]]; then
    usage
    exit 1
  fi

  local input_arg="" input_path=""
  if [[ $# -eq 0 ]]; then
    if ! input_path="$(find_latest_results_archive "$(pwd)")"; then
      echo "Error: no result*.tar.gz archive found in current dir: $(pwd)" >&2
      exit 1
    fi
    echo "No unpack input provided. Using latest archive: $input_path"
  else
    input_arg="$1"
    if [[ "$input_arg" = /* ]]; then
      input_path="$input_arg"
    else
      input_path="$(pwd)/$input_arg"
    fi

    if [[ "$input_arg" != *.tar.gz && ! "$input_arg" =~ \.part\.[0-9]{3}$ ]]; then
      local -a matching_parts=()
      shopt -s nullglob
      matching_parts=("${input_path}"[0-9][0-9][0-9])
      shopt -u nullglob

      if [[ ${#matching_parts[@]} -eq 0 ]]; then
        if ! input_path="$(find_latest_results_archive "$(pwd)")"; then
          echo "Error: input is not .tar.gz and no result*.tar.gz archive found in current dir: $(pwd)" >&2
          exit 1
        fi
        echo "Input '$input_arg' is not .tar.gz. Using latest archive: $input_path"
      fi
    fi
  fi

  local temp_archive=""
  local temp_extract_dir=""
  local archive_path=""
  local source_dir=""
  local -a source_archives=()

  cleanup() {
    if [[ -n "${temp_archive:-}" && -f "${temp_archive:-}" ]]; then
      rm -f "${temp_archive:-}"
    fi
    if [[ -n "${temp_extract_dir:-}" && -d "${temp_extract_dir:-}" ]]; then
      rm -rf "${temp_extract_dir:-}"
    fi
  }
  trap cleanup EXIT

  if [[ -f "$input_path" && "$input_path" == *.tar.gz ]]; then
    archive_path="$input_path"
    source_dir="$(dirname "$input_path")"
    source_archives=("$input_path")
  else
    local part_prefix
    if [[ -f "$input_path" && "$input_path" =~ \.part\.[0-9]{3}$ ]]; then
      part_prefix="${input_path%[0-9][0-9][0-9]}"
    else
      part_prefix="$input_path"
    fi

    local -a part_files
    shopt -s nullglob
    part_files=("${part_prefix}"[0-9][0-9][0-9])
    shopt -u nullglob

    if [[ ${#part_files[@]} -eq 0 ]]; then
      echo "Error: cannot find archive or split parts from input: $input_arg" >&2
      exit 1
    fi

    temp_archive="/tmp/results_sync_unpack_$(date +%s)_$$.tar.gz"
    cat "${part_files[@]}" > "$temp_archive"
    archive_path="$temp_archive"
    source_dir="$(dirname "${part_files[0]}")"
    source_archives=("${part_files[@]}")
    echo "Rebuilt archive from ${#part_files[@]} part files."
  fi

  temp_extract_dir="/tmp/results_sync_extract_$(date +%s)_$$"
  mkdir -p "$temp_extract_dir"

  local -a tar_extract_opts
  tar_extract_opts=()
  if tar_supports_flag "--warning"; then
    tar_extract_opts+=(--warning=no-unknown-keyword)
  fi

  if (( ${#tar_extract_opts[@]} > 0 )); then
    tar "${tar_extract_opts[@]}" -xzf "$archive_path" -C "$temp_extract_dir"
  else
    tar -xzf "$archive_path" -C "$temp_extract_dir"
  fi

  local extracted_grec_seqrec=""
  local extracted_genrec_results=""
  local candidate

  for candidate in \
    "$temp_extract_dir/GRec/results/test/seqrec-constrained" \
    "$temp_extract_dir/GRec_public/results/test/seqrec-constrained"; do
    if [[ -d "$candidate" ]]; then
      extracted_grec_seqrec="$candidate"
      break
    fi
  done

  for candidate in \
    "$temp_extract_dir/GenRec/results"; do
    if [[ -d "$candidate" ]]; then
      extracted_genrec_results="$candidate"
      break
    fi
  done

  if [[ -z "$extracted_grec_seqrec" ]]; then
    echo "Error: cannot find extracted GRec seqrec-constrained results in archive." >&2
    exit 1
  fi
  if [[ -z "$extracted_genrec_results" ]]; then
    echo "Error: cannot find extracted GenRec results in archive." >&2
    exit 1
  fi

  mkdir -p "$(dirname "$LOCAL_GREC_SEQREC_DIR")"
  mkdir -p "$(dirname "$LOCAL_GENREC_RESULTS_DIR")"
  rm -rf "$LOCAL_GREC_SEQREC_DIR" "$LOCAL_GENREC_RESULTS_DIR"

  mv "$extracted_grec_seqrec" "$LOCAL_GREC_SEQREC_DIR"
  mv "$extracted_genrec_results" "$LOCAL_GENREC_RESULTS_DIR"

  local grec_files genrec_files
  grec_files=$(find "$LOCAL_GREC_SEQREC_DIR" -type f | wc -l | tr -d '[:space:]')
  genrec_files=$(find "$LOCAL_GENREC_RESULTS_DIR" -type f | wc -l | tr -d '[:space:]')

  echo "Unpack complete."
  echo "Overwritten local dirs:"
  echo "  - $LOCAL_GREC_SEQREC_DIR (files: $grec_files)"
  echo "  - $LOCAL_GENREC_RESULTS_DIR (files: $genrec_files)"

  local archive_file
  for archive_file in "${source_archives[@]:-}"; do
    [[ -f "$archive_file" ]] && rm -f "$archive_file"
  done

  if [[ -n "$source_dir" ]]; then
    local -a old_results_archives
    shopt -s nullglob
    old_results_archives=(
      "$source_dir"/results_*.tar.gz
      "$source_dir"/results_*.tar.gz.part.[0-9][0-9][0-9]
    )
    shopt -u nullglob

    if [[ ${#old_results_archives[@]} -gt 0 ]]; then
      rm -f "${old_results_archives[@]}"
      echo "Removed results archive file(s) in source dir: ${#old_results_archives[@]}"
    fi
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  pack)
    shift
    pack_results "$@"
    ;;
  unpack)
    shift
    unpack_results "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Error: unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
