#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

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

DATASET="${DATASET:-Instruments}"
DATA_PATH="${DATA_PATH:-./data}"
BASE_MODEL="${BASE_MODEL:-ckpt/base_model/llava-hf/llava-onevision-qwen2-7b-ov-hf}"
MODEL_TYPE="${MODEL_TYPE:-llava_onevision}"
INDEX_FILE="${INDEX_FILE:-.index_qwen7B.json}"
INDEX_KEY="${INDEX_FILE#.}"
INDEX_KEY="${INDEX_KEY%.json}"
INDEX_KEY="${INDEX_KEY//\//_}"
DATASET_TAG="${DATASET//,/-}"
OUTPUT_DIR="${OUTPUT_DIR:-./ckpt/${DATASET_TAG}/llava-onevision-sft__idx-${INDEX_KEY}}"

CHECK_INDEX_FILES="${CHECK_INDEX_FILES:-true}"
export WANDB_NAME="${WANDB_NAME:-sft_vl_${DATASET_TAG}__idx-${INDEX_KEY}}"

GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-33325}"

SEED="${SEED:-42}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-2}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
EPOCHS="${EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_AND_EVAL_STRATEGY="${SAVE_AND_EVAL_STRATEGY:-epoch}"
SAVE_AND_EVAL_STEPS="${SAVE_AND_EVAL_STEPS:-1000}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./config/ds_z2_bf16.json}"

TASKS="${TASKS:-item2index,seqrec,index2item,fusionseqrec}"
TRAIN_PROMPT_SAMPLE_NUM="${TRAIN_PROMPT_SAMPLE_NUM:-1,1,1,1}"
TRAIN_DATA_SAMPLE_NUM="${TRAIN_DATA_SAMPLE_NUM:-0,0,0,0}"
RATIO_DATASET="${RATIO_DATASET:-1}"

USE_LORA="${USE_LORA:-false}"
LORA_MODULES_TO_SAVE="${LORA_MODULES_TO_SAVE:-embed_tokens,lm_head}"
FREEZE="${FREEZE:-visual}"
ONLY_TRAIN_RESPONSE="${ONLY_TRAIN_RESPONSE:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
REPORT_TO="${REPORT_TO:-wandb}"
DETERMINISTIC="${DETERMINISTIC:-false}"
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
    echo "[finetune/vl][WARN] Detected HOPE/AFO env but RUN_IN_FOREGROUND=false; forcing foreground to keep job alive."
    RUN_IN_FOREGROUND="true"
fi

if [[ "${CHECK_INDEX_FILES}" == "true" ]]; then
    IFS=',' read -r -a DATASET_LIST <<< "${DATASET}"
    for ds in "${DATASET_LIST[@]}"; do
        ds="${ds// /}"
        index_path="${DATA_PATH}/${ds}/${ds}${INDEX_FILE}"
        if [[ ! -f "${index_path}" ]]; then
            echo "[finetune/vl] Missing index file: ${index_path}" >&2
            echo "[finetune/vl] Hint: check DATA_PATH/DATASET/INDEX_FILE." >&2
            exit 1
        fi
    done
fi

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"
WANDB_DIR="${WANDB_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ "${WANDB_DIR%/}" == "${SCRIPT_REPO_ROOT%/}/wandb" ]]; then
    echo "[finetune/vl][WARN] WANDB_DIR points to .../wandb; normalize to repo root to avoid wandb/wandb nesting."
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
    --learning_rate "${LEARNING_RATE}"
    --epochs "${EPOCHS}"
    --weight_decay "${WEIGHT_DECAY}"
    --save_and_eval_strategy "${SAVE_AND_EVAL_STRATEGY}"
    --save_and_eval_steps "${SAVE_AND_EVAL_STEPS}"
    --deepspeed "${DEEPSPEED_CONFIG}"
    --bf16
    --tasks "${TASKS}"
    --train_prompt_sample_num "${TRAIN_PROMPT_SAMPLE_NUM}"
    --train_data_sample_num "${TRAIN_DATA_SAMPLE_NUM}"
    --ratio_dataset "${RATIO_DATASET}"
    --report_to "${REPORT_TO}"
    --index_file "${INDEX_FILE}"
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

echo "[finetune/vl] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[finetune/vl] LOG_FILE=${LOG_FILE}"
echo "[finetune/vl] MODEL_TYPE=${MODEL_TYPE} TASKS=${TASKS}"
echo "[finetune/vl] DATASET=${DATASET} INDEX_FILE=${INDEX_FILE} INDEX_KEY=${INDEX_KEY}"
echo "[finetune/vl] MANAGED_SCHEDULER_ENV=${MANAGED_SCHEDULER_ENV}"
echo "[finetune/vl] RUN_IN_FOREGROUND=${RUN_IN_FOREGROUND}"
echo "[finetune/vl] WANDB_DIR=${WANDB_DIR}"

if [[ "${DEBUG}" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="${DEBUG_GPU:-0}"
    python -m src.finetune.train_ddp_vl "${COMMON_ARGS[@]}" --debug
else
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    if [[ "${RUN_IN_FOREGROUND}" == "true" ]]; then
        echo "[finetune/vl] Starting DDP in foreground (torchrun)"
        echo "[finetune/vl] Foreground log will be tee'd to: ${LOG_FILE}"
        torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
            -m src.finetune.train_ddp_vl "${COMMON_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    else
        nohup torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
            -m src.finetune.train_ddp_vl "${COMMON_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
        PID=$!
        echo "Training started with PID=${PID}"
        echo "${PID}" >"${OUTPUT_DIR}/training.pid"
        echo "tail -f ${LOG_FILE}"
    fi
fi
