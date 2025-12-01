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

export WANDB_MODE=offline
export WANDB_ENTITY=wncfht
export WANDB_PROJECT=GRec_rl
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DATASET=Instruments
OUTPUT_DIR=ckpt/$DATASET/llava_rl
DATA_PATH=./data

export WANDB_NAME=llava_rl_1130
INDEX_FILE=.index_qwen7B.json
TASK=seqrec
# BASE_MODEL=ckpt/Instruments/Llava-onevision-finetune-item2index-seqrec-fusionseqrec/checkpoint-4098
BASE_MODEL=ckpt/Instruments/llava_rl/checkpoint-1593
MODEL_TYPE=llava_onevision

CHECKPOINT_NAME=$(basename "$BASE_MODEL")
MODEL_DIR_NAME=$(basename "$(dirname "$BASE_MODEL")")
LOG_FILE="log/${MODEL_DIR_NAME}-${CHECKPOINT_NAME}-${TASK}-${TIMESTAMP}.log"

COMMON_ARGS=(
    --model_type "$MODEL_TYPE"
    --base_model "$BASE_MODEL"
    --train_batch_size 64
    --eval_batch_size 128
    --num_train_epochs 2
    --gradient_accumulation_steps 2
    --eval_step 0.0999
    --reward_type ranking
    --num_generations 16
    --beam_search
    --temperature 1.0
    --learning_rate 1e-5
    --beta 1e-3
    --data_path "$DATA_PATH"
    --dataset "$DATASET"
    --index_file "$INDEX_FILE"
    --output_dir "$OUTPUT_DIR"
    --tasks "$TASK"
    --train_prompt_sample_num 1
    --train_data_sample_num 0
    --bf16
)

RUN_ARGS=("${COMMON_ARGS[@]}")

if $DEBUG; then
    export CUDA_VISIBLE_DEVICES=0
    python -m src.rl.rl_new "${RUN_ARGS[@]}"
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    mkdir -p log
    nohup accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 4 --main_process_port 29503 \
        --module src.rl.rl_new "${RUN_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "Training started with PID: $PID"
    echo "To stop training: kill $PID"
    echo "$LOG_FILE"
fi
