#!/bin/bash
set -euo pipefail

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

export WANDB_LOG_MODEL=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-wncfht}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRec_rl}"
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DATASET="${DATASET:-Instruments}"
DATA_PATH="${DATA_PATH:-./data}"
BASE_MODEL="${BASE_MODEL:-ckpt/Instruments/Llava-onevision-finetune-item2index-seqrec-fusionseqrec/checkpoint-4098}"
MODEL_TYPE="${MODEL_TYPE:-llava_onevision}"
INDEX_FILE="${INDEX_FILE:-.index_qwen7B.json}"
INDEX_KEY="${INDEX_FILE#.}"
INDEX_KEY="${INDEX_KEY%.json}"
INDEX_KEY="${INDEX_KEY//\//_}"
DATASET_TAG="${DATASET//,/-}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpt/${DATASET_TAG}/rl_default__idx-${INDEX_KEY}}"

CHECK_INDEX_FILES="${CHECK_INDEX_FILES:-true}"
export WANDB_NAME="${WANDB_NAME:-rl_${DATASET_TAG}__idx-${INDEX_KEY}}"

TASK="${TASK:-seqrec}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
GRAD_ACC="${GRAD_ACC:-2}"
EVAL_STEP="${EVAL_STEP:-0.0999}"
REWARD_TYPE="${REWARD_TYPE:-ranking}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BETA="${BETA:-1e-3}"
TRAIN_PROMPT_SAMPLE_NUM="${TRAIN_PROMPT_SAMPLE_NUM:-1}"
TRAIN_DATA_SAMPLE_NUM="${TRAIN_DATA_SAMPLE_NUM:-0}"
BF16="${BF16:-true}"

USE_BEAM_SEARCH="${USE_BEAM_SEARCH:-true}"
TEST_DURING_TRAINING="${TEST_DURING_TRAINING:-true}"
EVAL_ON_TEST="${EVAL_ON_TEST:-true}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"
COMPLETION_LOG_INTERVAL="${COMPLETION_LOG_INTERVAL:-100}"
DETERMINISTIC="${DETERMINISTIC:-false}"

GPUS="${GPUS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-./config/zero2_opt.yaml}"

if [[ "${CHECK_INDEX_FILES}" == "true" ]]; then
    IFS=',' read -r -a DATASET_LIST <<< "${DATASET}"
    for ds in "${DATASET_LIST[@]}"; do
        ds="${ds// /}"
        index_path="${DATA_PATH}/${ds}/${ds}${INDEX_FILE}"
        if [[ ! -f "${index_path}" ]]; then
            echo "[rl] Missing index file: ${index_path}" >&2
            echo "[rl] Hint: check DATA_PATH/DATASET/INDEX_FILE." >&2
            exit 1
        fi
    done
fi

mkdir -p log "${OUTPUT_DIR}"
CHECKPOINT_NAME=$(basename "${BASE_MODEL}")
MODEL_DIR_NAME=$(basename "$(dirname "${BASE_MODEL}")")
LOG_FILE="log/${MODEL_DIR_NAME}-${CHECKPOINT_NAME}-${TASK}-${INDEX_KEY}-${TIMESTAMP}.log"

COMMON_ARGS=(
    --model_type "${MODEL_TYPE}"
    --base_model "${BASE_MODEL}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --gradient_accumulation_steps "${GRAD_ACC}"
    --eval_step "${EVAL_STEP}"
    --reward_type "${REWARD_TYPE}"
    --num_generations "${NUM_GENERATIONS}"
    --temperature "${TEMPERATURE}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --learning_rate "${LEARNING_RATE}"
    --beta "${BETA}"
    --data_path "${DATA_PATH}"
    --dataset "${DATASET}"
    --index_file "${INDEX_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --tasks "${TASK}"
    --train_prompt_sample_num "${TRAIN_PROMPT_SAMPLE_NUM}"
    --train_data_sample_num "${TRAIN_DATA_SAMPLE_NUM}"
)

if [[ "${BF16}" == "true" ]]; then
    COMMON_ARGS+=(--bf16)
fi

if [[ "${USE_BEAM_SEARCH}" == "true" ]]; then
    COMMON_ARGS+=(--beam_search)
fi

if [[ "${TEST_DURING_TRAINING}" == "true" ]]; then
    COMMON_ARGS+=(--test_during_training)
fi

if [[ "${EVAL_ON_TEST}" == "true" ]]; then
    COMMON_ARGS+=(--eval_on_test)
fi

if [[ "${LOG_COMPLETIONS}" == "true" ]]; then
    COMMON_ARGS+=(--log_completions --completion_log_interval "${COMPLETION_LOG_INTERVAL}")
fi

if [[ "${DETERMINISTIC}" == "true" ]]; then
    COMMON_ARGS+=(--deterministic)
fi

RUN_ARGS=("${COMMON_ARGS[@]}")

echo "[rl] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[rl] LOG_FILE=${LOG_FILE}"
echo "[rl] MODEL_TYPE=${MODEL_TYPE} TASK=${TASK} REWARD_TYPE=${REWARD_TYPE}"
echo "[rl] DATASET=${DATASET} INDEX_FILE=${INDEX_FILE} INDEX_KEY=${INDEX_KEY}"

if $DEBUG; then
    export CUDA_VISIBLE_DEVICES="${DEBUG_GPU:-0}"
    python -m src.rl.rl "${RUN_ARGS[@]}"
else
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    nohup accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}" \
        --module src.rl.rl "${RUN_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "Training started with PID: $PID"
    echo "To stop training: kill $PID"
    echo "$LOG_FILE"
fi
