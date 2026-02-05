#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Generate cluster-friendly runnable scripts for SFT/RL with stable naming.

Usage:
  bash scripts/tools/gen_run.sh --stage <sft_text|sft_vl|rl> [options]

Core options:
  --stage <value>                 Required. One of: sft_text, sft_vl, rl
  --dataset <value>               Dataset name (default: Instruments)
  --tasks <value>                 Task list, comma-separated
  --base-model <value>            Base model path
  --model-type <value>            model_type argument
  --index-file <value>            Index suffix file, e.g. .index_qwen7B.json
  --seed <int>                    Random seed (default: 42)

Naming/output options:
  --name-suffix <value>           Extra suffix appended to RUN_KEY
  --output-root <path>            Root for ckpt output (default: ckpt)
  --log-root <path>               Root for logs (default: log)
  --runs-root <path>              Root for generated scripts/spec/meta (default: runs)

Runtime options:
  --data-path <path>              Data root (default: ./data)
  --gpus <csv>                    CUDA_VISIBLE_DEVICES (default: 0,1,2,3)
  --nproc <int>                   torchrun nproc_per_node (SFT only)
  --master-port <int>             torchrun master_port (SFT only)
  --num-processes <int>           accelerate num_processes (RL only)
  --main-process-port <int>       accelerate main_process_port (RL only)
  --accelerate-config <path>      accelerate config file (RL only)

Wandb options:
  --wandb-project <value>         WANDB_PROJECT (default by stage)
  --wandb-mode <value>            WANDB_MODE (default: offline)
  --wandb-entity <value>          WANDB_ENTITY (default: empty)

Advanced options:
  --set KEY=VALUE                 Override any variable (can repeat)
                                  Example: --set LEARNING_RATE=1e-5 --set EPOCHS=4
  --help                          Show help

Examples:
  bash scripts/tools/gen_run.sh --stage sft_text \
    --dataset Instruments \
    --tasks item2index,seqrec \
    --base-model ckpt/base_model/Qwen2.5-3B-Instruct \
    --model-type qwen2_5_instruct \
    --index-file .index_qwen3-embedding-4B.json \
    --set PER_DEVICE_BATCH_SIZE=8 --set GRAD_ACC=4

  bash scripts/tools/gen_run.sh --stage rl \
    --dataset Instruments \
    --tasks seqrec \
    --base-model ckpt/Instruments/qwen2.5-sft/checkpoint-1234 \
    --model-type qwen2_5_instruct \
    --index-file .index_qwen3-embedding-4B.json \
    --set REWARD_TYPE=ranking --set NUM_GENERATIONS=16
USAGE
}

die() {
    echo "[gen_run] $*" >&2
    exit 1
}

slugify() {
    local input="$1"
    local slug
    slug=$(printf '%s' "$input" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')
    if [[ -z "$slug" ]]; then
        slug="na"
    fi
    printf '%s' "$slug"
}

shorten_with_hash() {
    local value="$1"
    local max_len="$2"

    if (( ${#value} <= max_len )); then
        printf '%s' "$value"
        return
    fi

    local digest
    digest=$(printf '%s' "$value" | shasum | awk '{print $1}' | cut -c1-10)
    local head_len=$((max_len - 13))
    if (( head_len < 1 )); then
        head_len=1
    fi

    printf '%s__h%s' "${value:0:head_len}" "$digest"
}

is_true() {
    local value
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$value" in
        true|1|yes|y|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

need_value() {
    local flag="$1"
    local value="$2"
    if [[ -z "$value" ]]; then
        die "Missing value for ${flag}"
    fi
}

STAGE=""
DATASET=""
TASKS=""
BASE_MODEL=""
MODEL_TYPE=""
INDEX_FILE=""
SEED=""
NAME_SUFFIX=""
OUTPUT_ROOT=""
LOG_ROOT=""
RUNS_ROOT=""
DATA_PATH=""
GPUS=""
NPROC=""
MASTER_PORT=""
NUM_PROCESSES=""
MAIN_PROCESS_PORT=""
ACCELERATE_CONFIG=""
WANDB_PROJECT=""
WANDB_MODE=""
WANDB_ENTITY=""

USER_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="${2:-}"
            need_value "$1" "$STAGE"
            shift 2
            ;;
        --dataset)
            DATASET="${2:-}"
            need_value "$1" "$DATASET"
            shift 2
            ;;
        --tasks)
            TASKS="${2:-}"
            need_value "$1" "$TASKS"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="${2:-}"
            need_value "$1" "$BASE_MODEL"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="${2:-}"
            need_value "$1" "$MODEL_TYPE"
            shift 2
            ;;
        --index-file)
            INDEX_FILE="${2:-}"
            need_value "$1" "$INDEX_FILE"
            shift 2
            ;;
        --seed)
            SEED="${2:-}"
            need_value "$1" "$SEED"
            shift 2
            ;;
        --name-suffix)
            NAME_SUFFIX="${2:-}"
            need_value "$1" "$NAME_SUFFIX"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="${2:-}"
            need_value "$1" "$OUTPUT_ROOT"
            shift 2
            ;;
        --log-root)
            LOG_ROOT="${2:-}"
            need_value "$1" "$LOG_ROOT"
            shift 2
            ;;
        --runs-root)
            RUNS_ROOT="${2:-}"
            need_value "$1" "$RUNS_ROOT"
            shift 2
            ;;
        --data-path)
            DATA_PATH="${2:-}"
            need_value "$1" "$DATA_PATH"
            shift 2
            ;;
        --gpus)
            GPUS="${2:-}"
            need_value "$1" "$GPUS"
            shift 2
            ;;
        --nproc)
            NPROC="${2:-}"
            need_value "$1" "$NPROC"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="${2:-}"
            need_value "$1" "$MASTER_PORT"
            shift 2
            ;;
        --num-processes)
            NUM_PROCESSES="${2:-}"
            need_value "$1" "$NUM_PROCESSES"
            shift 2
            ;;
        --main-process-port)
            MAIN_PROCESS_PORT="${2:-}"
            need_value "$1" "$MAIN_PROCESS_PORT"
            shift 2
            ;;
        --accelerate-config)
            ACCELERATE_CONFIG="${2:-}"
            need_value "$1" "$ACCELERATE_CONFIG"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="${2:-}"
            need_value "$1" "$WANDB_PROJECT"
            shift 2
            ;;
        --wandb-mode)
            WANDB_MODE="${2:-}"
            need_value "$1" "$WANDB_MODE"
            shift 2
            ;;
        --wandb-entity)
            WANDB_ENTITY="${2:-}"
            need_value "$1" "$WANDB_ENTITY"
            shift 2
            ;;
        --set)
            if [[ $# -lt 2 ]]; then
                die "Missing KEY=VALUE after --set"
            fi
            if [[ "$2" != *=* ]]; then
                die "Invalid --set format: $2 (expected KEY=VALUE)"
            fi
            export "$2"
            USER_OVERRIDES+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

if [[ -z "$STAGE" ]]; then
    die "--stage is required"
fi

case "$STAGE" in
    sft_text|sft_vl|rl)
        ;;
    *)
        die "Invalid --stage: $STAGE (expected sft_text|sft_vl|rl)"
        ;;
esac

DATASET="${DATASET:-Instruments}"
DATA_PATH="${DATA_PATH:-./data}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ckpt}"
LOG_ROOT="${LOG_ROOT:-log}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
GPUS="${GPUS:-0,1,2,3}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

if [[ "$STAGE" == "sft_text" ]]; then
    BASE_MODEL="${BASE_MODEL:-ckpt/base_model/Qwen2.5-3B-Instruct}"
    MODEL_TYPE="${MODEL_TYPE:-qwen2_5_instruct}"
    TASKS="${TASKS:-item2index,seqrec}"
    INDEX_FILE="${INDEX_FILE:-.index_qwen3-embedding-4B.json}"
    WANDB_PROJECT="${WANDB_PROJECT:-GRec-sft}"

    NPROC="${NPROC:-4}"
    MASTER_PORT="${MASTER_PORT:-33326}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
    GRAD_ACC="${GRAD_ACC:-4}"
    NUM_WORKERS="${NUM_WORKERS:-16}"
    LEARNING_RATE="${LEARNING_RATE:-5e-5}"
    EPOCHS="${EPOCHS:-2}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
    LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
    SAVE_AND_EVAL_STRATEGY="${SAVE_AND_EVAL_STRATEGY:-epoch}"
    SAVE_AND_EVAL_STEPS="${SAVE_AND_EVAL_STEPS:-1000}"
    DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./config/ds_z2_bf16.json}"
    TRAIN_PROMPT_SAMPLE_NUM="${TRAIN_PROMPT_SAMPLE_NUM:-1,1}"
    TRAIN_DATA_SAMPLE_NUM="${TRAIN_DATA_SAMPLE_NUM:-0,0}"
    RATIO_DATASET="${RATIO_DATASET:-1}"

    USE_LORA="${USE_LORA:-false}"
    LORA_MODULES_TO_SAVE="${LORA_MODULES_TO_SAVE:-embed_tokens,lm_head}"
    FREEZE="${FREEZE:-}"
    ONLY_TRAIN_RESPONSE="${ONLY_TRAIN_RESPONSE:-true}"
    USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
    REPORT_TO="${REPORT_TO:-wandb}"
    DETERMINISTIC="${DETERMINISTIC:-false}"
    EVAL_BY_DATASET="${EVAL_BY_DATASET:-false}"
    EVAL_MAIN_DATASET="${EVAL_MAIN_DATASET:-}"
    BF16="${BF16:-true}"
elif [[ "$STAGE" == "sft_vl" ]]; then
    BASE_MODEL="${BASE_MODEL:-ckpt/base_model/llava-hf/llava-onevision-qwen2-7b-ov-hf}"
    MODEL_TYPE="${MODEL_TYPE:-llava_onevision}"
    TASKS="${TASKS:-item2index,seqrec,index2item,fusionseqrec}"
    INDEX_FILE="${INDEX_FILE:-.index_qwen7B.json}"
    WANDB_PROJECT="${WANDB_PROJECT:-GRec-sft}"

    NPROC="${NPROC:-4}"
    MASTER_PORT="${MASTER_PORT:-33325}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
    GRAD_ACC="${GRAD_ACC:-2}"
    NUM_WORKERS="${NUM_WORKERS:-16}"
    LEARNING_RATE="${LEARNING_RATE:-5e-5}"
    EPOCHS="${EPOCHS:-5}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
    SAVE_AND_EVAL_STRATEGY="${SAVE_AND_EVAL_STRATEGY:-epoch}"
    SAVE_AND_EVAL_STEPS="${SAVE_AND_EVAL_STEPS:-1000}"
    DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./config/ds_z2_bf16.json}"
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
    BF16="${BF16:-true}"
else
    BASE_MODEL="${BASE_MODEL:-ckpt/Instruments/Llava-onevision-finetune-item2index-seqrec-fusionseqrec/checkpoint-4098}"
    MODEL_TYPE="${MODEL_TYPE:-llava_onevision}"
    TASKS="${TASKS:-seqrec}"
    INDEX_FILE="${INDEX_FILE:-.index_qwen7B.json}"
    WANDB_PROJECT="${WANDB_PROJECT:-GRec-rl}"

    NUM_PROCESSES="${NUM_PROCESSES:-4}"
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-./config/zero2_opt.yaml}"

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
fi

DATE_TAG=$(date +"%Y%m%d")
TIME_TAG=$(date +"%Y%m%d-%H%M%S")
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
JOB_RAW="${SLURM_JOB_ID:-${PBS_JOBID:-local}}"
JOB_TAG="j$(slugify "$JOB_RAW")"

DATASET_SLUG=$(slugify "$DATASET")
TASKS_SLUG=$(slugify "$TASKS")
MODEL_SLUG=$(slugify "$(basename "$BASE_MODEL")")
INDEX_NAME="$(basename "$INDEX_FILE")"
INDEX_NAME="${INDEX_NAME%.json}"
INDEX_NAME="${INDEX_NAME#.}"
INDEX_SLUG=$(slugify "$INDEX_NAME")

RUN_KEY="${STAGE}__${DATASET_SLUG}__${TASKS_SLUG}__${MODEL_SLUG}__idx-${INDEX_SLUG}__s${SEED}"
if [[ -n "$NAME_SUFFIX" ]]; then
    RUN_KEY="${RUN_KEY}__$(slugify "$NAME_SUFFIX")"
fi
RUN_KEY=$(shorten_with_hash "$RUN_KEY" 170)

RUN_ID="${RUN_KEY}__${TIME_TAG}__g${GIT_SHA}__${JOB_TAG}"
RUN_ID=$(shorten_with_hash "$RUN_ID" 220)

OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_SLUG}/${STAGE}/${RUN_ID}"
LOG_FILE="${LOG_ROOT}/${STAGE}/${DATASET_SLUG}/${DATE_TAG}/${RUN_ID}.log"

GENERATED_DIR="${RUNS_ROOT}/generated/${STAGE}/${DATE_TAG}"
SPEC_DIR="${RUNS_ROOT}/specs/${STAGE}/${DATE_TAG}"
META_DIR="${RUNS_ROOT}/meta/${STAGE}/${DATE_TAG}"
SCRIPT_PATH="${GENERATED_DIR}/${RUN_ID}.sh"
SPEC_PATH="${SPEC_DIR}/${RUN_ID}.env"
META_PATH="${META_DIR}/${RUN_ID}.txt"

mkdir -p "$GENERATED_DIR" "$SPEC_DIR" "$META_DIR"

RUN_CMD=()
if [[ "$STAGE" == "sft_text" || "$STAGE" == "sft_vl" ]]; then
    MODULE="src.finetune.train_ddp"
    if [[ "$STAGE" == "sft_vl" ]]; then
        MODULE="src.finetune.train_ddp_vl"
    fi

    RUN_CMD=(
        torchrun
        --nproc_per_node "$NPROC"
        --master_port "$MASTER_PORT"
        -m "$MODULE"
        --seed "$SEED"
        --base_model "$BASE_MODEL"
        --model_type "$MODEL_TYPE"
        --output_dir "$OUTPUT_DIR"
        --dataset "$DATASET"
        --data_path "$DATA_PATH"
        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
        --gradient_accumulation_steps "$GRAD_ACC"
        --num_workers "$NUM_WORKERS"
        --learning_rate "$LEARNING_RATE"
        --epochs "$EPOCHS"
        --save_and_eval_strategy "$SAVE_AND_EVAL_STRATEGY"
        --save_and_eval_steps "$SAVE_AND_EVAL_STEPS"
        --weight_decay "$WEIGHT_DECAY"
        --deepspeed "$DEEPSPEED_CONFIG"
        --tasks "$TASKS"
        --train_prompt_sample_num "$TRAIN_PROMPT_SAMPLE_NUM"
        --train_data_sample_num "$TRAIN_DATA_SAMPLE_NUM"
        --ratio_dataset "$RATIO_DATASET"
        --index_file "$INDEX_FILE"
        --report_to "$REPORT_TO"
    )

    if [[ "$STAGE" == "sft_text" ]]; then
        RUN_CMD+=(--lr_scheduler_type "$LR_SCHEDULER_TYPE")
    fi

    if is_true "$BF16"; then
        RUN_CMD+=(--bf16)
    fi
    if is_true "$USE_GRADIENT_CHECKPOINTING"; then
        RUN_CMD+=(--use_gradient_checkpointing)
    fi
    if is_true "$ONLY_TRAIN_RESPONSE"; then
        RUN_CMD+=(--only_train_response)
    fi
    if is_true "$USE_LORA"; then
        RUN_CMD+=(--use_lora --lora_modules_to_save "$LORA_MODULES_TO_SAVE")
    fi
    if [[ -n "$FREEZE" ]]; then
        RUN_CMD+=(--freeze "$FREEZE")
    fi
    if is_true "$DETERMINISTIC"; then
        RUN_CMD+=(--deterministic)
    fi
    if [[ "$STAGE" == "sft_text" ]] && is_true "$EVAL_BY_DATASET"; then
        RUN_CMD+=(--eval_by_dataset)
        if [[ -n "$EVAL_MAIN_DATASET" ]]; then
            RUN_CMD+=(--eval_main_dataset "$EVAL_MAIN_DATASET")
        fi
    fi
else
    RUN_CMD=(
        accelerate
        launch
        --config_file "$ACCELERATE_CONFIG"
        --num_processes "$NUM_PROCESSES"
        --main_process_port "$MAIN_PROCESS_PORT"
        --module src.rl.rl
        --seed "$SEED"
        --model_type "$MODEL_TYPE"
        --base_model "$BASE_MODEL"
        --train_batch_size "$TRAIN_BATCH_SIZE"
        --eval_batch_size "$EVAL_BATCH_SIZE"
        --num_train_epochs "$NUM_TRAIN_EPOCHS"
        --gradient_accumulation_steps "$GRAD_ACC"
        --eval_step "$EVAL_STEP"
        --reward_type "$REWARD_TYPE"
        --num_generations "$NUM_GENERATIONS"
        --temperature "$TEMPERATURE"
        --max_completion_length "$MAX_COMPLETION_LENGTH"
        --learning_rate "$LEARNING_RATE"
        --beta "$BETA"
        --data_path "$DATA_PATH"
        --dataset "$DATASET"
        --index_file "$INDEX_FILE"
        --output_dir "$OUTPUT_DIR"
        --tasks "$TASKS"
        --train_prompt_sample_num "$TRAIN_PROMPT_SAMPLE_NUM"
        --train_data_sample_num "$TRAIN_DATA_SAMPLE_NUM"
    )

    if is_true "$BF16"; then
        RUN_CMD+=(--bf16)
    fi
    if is_true "$USE_BEAM_SEARCH"; then
        RUN_CMD+=(--beam_search)
    fi
    if is_true "$TEST_DURING_TRAINING"; then
        RUN_CMD+=(--test_during_training)
    fi
    if is_true "$EVAL_ON_TEST"; then
        RUN_CMD+=(--eval_on_test)
    else
        RUN_CMD+=(--no_eval_on_test)
    fi
    if is_true "$LOG_COMPLETIONS"; then
        RUN_CMD+=(--log_completions --completion_log_interval "$COMPLETION_LOG_INTERVAL")
    fi
    if is_true "$DETERMINISTIC"; then
        RUN_CMD+=(--deterministic)
    fi
fi

CMD_PREVIEW=$(printf '%q ' "${RUN_CMD[@]}")

{
    printf 'STAGE=%q\n' "$STAGE"
    printf 'RUN_KEY=%q\n' "$RUN_KEY"
    printf 'RUN_ID=%q\n' "$RUN_ID"
    printf 'DATASET=%q\n' "$DATASET"
    printf 'TASKS=%q\n' "$TASKS"
    printf 'BASE_MODEL=%q\n' "$BASE_MODEL"
    printf 'MODEL_TYPE=%q\n' "$MODEL_TYPE"
    printf 'INDEX_FILE=%q\n' "$INDEX_FILE"
    printf 'DATA_PATH=%q\n' "$DATA_PATH"
    printf 'SEED=%q\n' "$SEED"
    printf 'WANDB_PROJECT=%q\n' "$WANDB_PROJECT"
    printf 'WANDB_MODE=%q\n' "$WANDB_MODE"
    printf 'WANDB_ENTITY=%q\n' "$WANDB_ENTITY"
    printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
    printf 'LOG_FILE=%q\n' "$LOG_FILE"
    printf 'GPUS=%q\n' "$GPUS"
    printf 'COMMAND=%q\n' "$CMD_PREVIEW"

    if (( ${#USER_OVERRIDES[@]} > 0 )); then
        echo ""
        echo "# user overrides"
        for item in "${USER_OVERRIDES[@]}"; do
            printf '%s\n' "$item"
        done
    fi
} > "$SPEC_PATH"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'run_key=%s\n' "$RUN_KEY"
    printf 'created_at=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    printf 'stage=%s\n' "$STAGE"
    printf 'git_sha=%s\n' "$GIT_SHA"
    printf 'user=%s\n' "${USER:-unknown}"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
    printf 'pbs_job_id=%s\n' "${PBS_JOBID:-}"
    printf 'generated_script=%s\n' "$SCRIPT_PATH"
    printf 'spec_file=%s\n' "$SPEC_PATH"
    printf 'command_preview=%s\n' "$CMD_PREVIEW"
} > "$META_PATH"

{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo
    printf 'RUN_KEY=%q\n' "$RUN_KEY"
    printf 'RUN_ID=%q\n' "$RUN_ID"
    printf 'STAGE=%q\n' "$STAGE"
    printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
    printf 'LOG_FILE=%q\n' "$LOG_FILE"
    echo
    printf 'export WANDB_MODE=%q\n' "$WANDB_MODE"
    printf 'export WANDB_PROJECT=%q\n' "$WANDB_PROJECT"
    if [[ -n "$WANDB_ENTITY" ]]; then
        printf 'export WANDB_ENTITY=%q\n' "$WANDB_ENTITY"
    fi
    printf 'export WANDB_NAME=%q\n' "$RUN_ID"
    printf 'export WANDB_GROUP=%q\n' "$RUN_KEY"
    echo 'export PYTHONUNBUFFERED=1'
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$GPUS"
    if [[ "$STAGE" == "rl" ]]; then
        echo 'export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"'
    fi
    echo
    echo 'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"'
    echo 'cd "${REPO_ROOT}"'
    echo 'mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}"'
    echo 'exec > >(tee -a "${LOG_FILE}") 2>&1'
    echo 'echo "[run] stage=${STAGE} run_id=${RUN_ID}"'
    echo 'echo "[run] output_dir=${OUTPUT_DIR}"'
    echo 'echo "[run] log_file=${LOG_FILE}"'
    echo
    echo 'CMD=('
    for token in "${RUN_CMD[@]}"; do
        printf '  %q\n' "$token"
    done
    echo ')'
    echo 'printf "%q " "${CMD[@]}" > "${OUTPUT_DIR}/cmd.sh"'
    echo 'echo >> "${OUTPUT_DIR}/cmd.sh"'
    echo 'chmod +x "${OUTPUT_DIR}/cmd.sh"'
    echo '"${CMD[@]}"'
} > "$SCRIPT_PATH"

chmod +x "$SCRIPT_PATH"

echo "[gen_run] Generated script: $SCRIPT_PATH"
echo "[gen_run] Spec file      : $SPEC_PATH"
echo "[gen_run] Meta file      : $META_PATH"
echo "[gen_run] RUN_KEY        : $RUN_KEY"
echo "[gen_run] RUN_ID         : $RUN_ID"
echo "[gen_run] Output dir     : $OUTPUT_DIR"
echo "[gen_run] Log file       : $LOG_FILE"
echo "[gen_run] Submit example : sbatch --job-name \"$RUN_ID\" \"$SCRIPT_PATH\""
