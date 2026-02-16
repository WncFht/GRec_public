#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

normalize_model_tag() {
    local raw="$1"
    raw="$(basename "$raw")"
    raw="${raw,,}"
    raw="${raw// /-}"
    raw="${raw//\//-}"
    raw="${raw//_/-}"
    raw="$(echo "$raw" | sed -E 's/[^a-z0-9.-]+/-/g; s/-+/-/g; s/^-+//; s/-+$//')"
    printf '%s' "$raw"
}

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate grec
export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/grec/lib:$LD_LIBRARY_PATH
export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/grec/bin:$PATH
export CC=$CONDA_PREFIX/bin/conda-cc-with-crypt.sh
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export TRITON_CACHE_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/.cache/triton

DEBUG=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-grec_sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-generate_rec}"
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

GREC_ROOT="${GREC_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec}"
ROOT_DIR="${ROOT_DIR:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"

DATASET="${DATASET:-Instruments}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/ckpt/base_model/Qwen2.5-3B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5_instruct}"
TASKS="${TASKS:-item2index,seqrec,fusionseqrec}"
INDEX_FILE="${INDEX_FILE:-.index_qwen3-embedding-4B.json}"
INDEX_KEY="${INDEX_FILE#.}"
INDEX_KEY="${INDEX_KEY%.json}"
INDEX_KEY="${INDEX_KEY//\//_}"
DATASET_TAG="${DATASET//,/-}"
TASKS_TAG="${TASKS// /}"
TASKS_TAG="${TASKS_TAG//,/-}"
MODEL_TAG_SOURCE="${SFT_MODEL_TAG:-$BASE_MODEL}"
SFT_MODEL_TAG="$(normalize_model_tag "$MODEL_TAG_SOURCE")"
if [[ -z "$SFT_MODEL_TAG" ]]; then
    echo "[finetune/text] Failed to derive SFT_MODEL_TAG from BASE_MODEL=${BASE_MODEL}" >&2
    echo "[finetune/text] Hint: export SFT_MODEL_TAG explicitly (e.g., qwen2.5-7b-instruct)." >&2
    exit 1
fi
export SFT_MODEL_TAG
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/ckpt/${DATASET_TAG}/${SFT_MODEL_TAG}-sft__tasks-${TASKS_TAG}__idx-${INDEX_KEY}__rid-${RUN_ID}}"

CHECK_INDEX_FILES="${CHECK_INDEX_FILES:-true}"
export WANDB_NAME="${WANDB_NAME:-sft_text_${DATASET_TAG}__model-${SFT_MODEL_TAG}__tasks-${TASKS_TAG}__idx-${INDEX_KEY}}"

GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-33326}"

SEED="${SEED:-42}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
EPOCHS="${EPOCHS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
SAVE_AND_EVAL_STRATEGY="${SAVE_AND_EVAL_STRATEGY:-epoch}"
SAVE_AND_EVAL_STEPS="${SAVE_AND_EVAL_STEPS:-1000}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$GREC_ROOT/config/ds_z2_bf16.json}"

TRAIN_PROMPT_SAMPLE_NUM="${TRAIN_PROMPT_SAMPLE_NUM:-1,1,1}"
TRAIN_DATA_SAMPLE_NUM="${TRAIN_DATA_SAMPLE_NUM:-0,0,0}"
RATIO_DATASET="${RATIO_DATASET:-1}"

USE_LORA="${USE_LORA:-false}"
LORA_MODULES_TO_SAVE="${LORA_MODULES_TO_SAVE:-embed_tokens,lm_head}"
FREEZE="${FREEZE:-}"
ONLY_TRAIN_RESPONSE="${ONLY_TRAIN_RESPONSE:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
REPORT_TO="${REPORT_TO:-wandb}"
DETERMINISTIC="${DETERMINISTIC:-false}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-false}"

MANAGED_SCHEDULER_ENV=false
if [[ -n "${AFO_APPID:-}" || -n "${HOPE_RUN_ID:-}" || -n "${HOPE_RUNID:-}" || -n "${AFO_TASK_ID:-}" ]]; then
    MANAGED_SCHEDULER_ENV=true
fi

RUN_IN_FOREGROUND_DEFAULT="false"
if [[ "${MANAGED_SCHEDULER_ENV}" == "true" ]]; then
    RUN_IN_FOREGROUND_DEFAULT="true"
fi
RUN_IN_FOREGROUND="${RUN_IN_FOREGROUND:-$RUN_IN_FOREGROUND_DEFAULT}"
if is_truthy "${RUN_IN_FOREGROUND}"; then
    RUN_IN_FOREGROUND="true"
else
    RUN_IN_FOREGROUND="false"
fi

if [[ "${MANAGED_SCHEDULER_ENV}" == "true" && "${RUN_IN_FOREGROUND}" != "true" ]]; then
    echo "[finetune/text][WARN] Detected HOPE/AFO env but RUN_IN_FOREGROUND=false; forcing foreground to keep job alive."
    RUN_IN_FOREGROUND="true"
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ "${PYTHON_BIN}" == /* ]]; then
    PYTHON_BIN_PATH="${PYTHON_BIN}"
else
    PYTHON_BIN_PATH="$(command -v "${PYTHON_BIN}" || true)"
fi

if [[ -z "${PYTHON_BIN_PATH}" || ! -x "${PYTHON_BIN_PATH}" ]]; then
    echo "[finetune/text] Python launcher not found/executable: ${PYTHON_BIN}" >&2
    exit 1
fi

EVAL_BY_DATASET="${EVAL_BY_DATASET:-true}"
EVAL_MAIN_DATASET="${EVAL_MAIN_DATASET:-Instruments}"

if [[ "${CHECK_INDEX_FILES}" == "true" ]]; then
    IFS=',' read -r -a DATASET_LIST <<< "${DATASET}"
    for ds in "${DATASET_LIST[@]}"; do
        ds="${ds// /}"
        index_path="${DATA_PATH}/${ds}/${ds}${INDEX_FILE}"
        if [[ ! -f "${index_path}" ]]; then
            echo "[finetune/text] Missing index file: ${index_path}" >&2
            echo "[finetune/text] Hint: check DATA_PATH/DATASET/INDEX_FILE." >&2
            exit 1
        fi
    done
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "[finetune/text] BASE_MODEL not found: ${BASE_MODEL}" >&2
    echo "[finetune/text] Hint: export BASE_MODEL=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/ckpt/base_model/Qwen2.5-3B-Instruct" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"
WANDB_DIR="${WANDB_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ "${WANDB_DIR%/}" == "${SCRIPT_REPO_ROOT%/}/wandb" ]]; then
    echo "[finetune/text][WARN] WANDB_DIR points to .../wandb; normalize to repo root to avoid wandb/wandb nesting."
    WANDB_DIR="${SCRIPT_REPO_ROOT}"
fi
export WANDB_DIR
mkdir -p "${WANDB_DIR}/wandb"

COMMON_ARGS=(
    --seed "${SEED}"
    --base_model "${BASE_MODEL}"
    --model_type "${MODEL_TYPE}"
    --output_dir "${OUTPUT_DIR}"
    --dataset "${DATASET}"
    --data_path "${DATA_PATH}"
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACC}"
    --num_workers "${NUM_WORKERS}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --learning_rate "${LEARNING_RATE}"
    --epochs "${EPOCHS}"
    --save_and_eval_strategy "${SAVE_AND_EVAL_STRATEGY}"
    --save_and_eval_steps "${SAVE_AND_EVAL_STEPS}"
    --weight_decay "${WEIGHT_DECAY}"
    --deepspeed "${DEEPSPEED_CONFIG}"
    --bf16
    --tasks "${TASKS}"
    --train_prompt_sample_num "${TRAIN_PROMPT_SAMPLE_NUM}"
    --train_data_sample_num "${TRAIN_DATA_SAMPLE_NUM}"
    --ratio_dataset "${RATIO_DATASET}"
    --index_file "${INDEX_FILE}"
    --report_to "${REPORT_TO}"
)

if [[ "${USE_GRADIENT_CHECKPOINTING}" == "true" ]]; then
    COMMON_ARGS+=(--use_gradient_checkpointing)
fi

if [[ "${ONLY_TRAIN_RESPONSE}" == "true" ]]; then
    COMMON_ARGS+=(--only_train_response)
fi

if [[ "${USE_LORA}" == "true" ]]; then
    COMMON_ARGS+=(--use_lora --lora_modules_to_save "${LORA_MODULES_TO_SAVE}")
fi

if [[ -n "${FREEZE}" ]]; then
    COMMON_ARGS+=(--freeze "${FREEZE}")
fi

if [[ "${DETERMINISTIC}" == "true" ]]; then
    COMMON_ARGS+=(--deterministic)
fi

if [[ "${EVAL_BY_DATASET}" == "true" ]]; then
    COMMON_ARGS+=(--eval_by_dataset)
    if [[ -n "${EVAL_MAIN_DATASET}" ]]; then
        COMMON_ARGS+=(--eval_main_dataset "${EVAL_MAIN_DATASET}")
    fi
fi

echo "[finetune/text] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[finetune/text] LOG_FILE=${LOG_FILE}"
echo "[finetune/text] MODEL_TYPE=${MODEL_TYPE} TASKS=${TASKS}"
echo "[finetune/text] SFT_MODEL_TAG=${SFT_MODEL_TAG} WANDB_NAME=${WANDB_NAME}"
echo "[finetune/text] DATASET=${DATASET} INDEX_FILE=${INDEX_FILE} INDEX_KEY=${INDEX_KEY}"
echo "[finetune/text] USE_TORCH_COMPILE=${USE_TORCH_COMPILE}"
echo "[finetune/text] MANAGED_SCHEDULER_ENV=${MANAGED_SCHEDULER_ENV}"
echo "[finetune/text] RUN_IN_FOREGROUND=${RUN_IN_FOREGROUND}"
echo "[finetune/text] WANDB_DIR=${WANDB_DIR}"
echo "[finetune/text] PYTHON_BIN=${PYTHON_BIN_PATH}"
echo "[finetune/text] PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"

export USE_TORCH_COMPILE

if [[ "${DEBUG}" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="${DEBUG_GPU:-0}"
    "${PYTHON_BIN_PATH}" -m src.finetune.train_ddp "${COMMON_ARGS[@]}" --debug
else
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    if [[ "${RUN_IN_FOREGROUND}" == "true" ]]; then
        echo "[finetune/text] Starting DDP in foreground (${PYTHON_BIN_PATH} -m torch.distributed.run)"
        echo "[finetune/text] Foreground log will be tee'd to: ${LOG_FILE}"
        "${PYTHON_BIN_PATH}" -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
            -m src.finetune.train_ddp "${COMMON_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    else
        nohup "${PYTHON_BIN_PATH}" -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
            -m src.finetune.train_ddp "${COMMON_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
        PID=$!
        echo "Training started with PID=${PID}"
        echo "${PID}" >"${OUTPUT_DIR}/training.pid"
        echo "tail -f ${LOG_FILE}"
    fi
fi
