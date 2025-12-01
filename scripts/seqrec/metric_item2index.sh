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

export CUDA_VISIBLE_DEVICES=0

# CKPT_PATH=ckpt/Instruments/Llava-onevision-finetune-item2index-index2item-seqrec-fusionseqrec-1-qwen7B/checkpoint-5619
# BASE_MODEL=./ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
# MODEL_TYPE=llava_onevision

# CKPT_PATH=ckpt/Instruments/Qwen2.5-7B-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463
# BASE_MODEL=ckpt/base_model/Qwen2.5-7B
# MODEL_TYPE=qwen

CKPT_PATH=/opt/meituan/dolphinfs_zhangkangning02/zkn/verl/checkpoints/grec_verl/qwen2-vl-7b-n32/global_step_105/actor/merged
BASE_MODEL=ckpt/base_model/Qwen2-VL-7B-Instruct
MODEL_TYPE=qwen2_vl

# DATASET=Arts,Automotive,Cell,Games,Instruments,Pet,Tools,Toys,Sports
# DATASET=Arts,Games,Instruments
DATASET=Instruments
TASK=item2index

RESULTS_DIR=./results
CHECKPOINT_NAME=$(basename "$CKPT_PATH")
MODEL_DIR_NAME=$(basename "$(dirname "$CKPT_PATH")")
RESULTS_FILE=${RESULTS_DIR}/${TASK}-${MODEL_DIR_NAME}-${CHECKPOINT_NAME}.txt
LOG_FILE=${RESULTS_FILE%.txt}_log.txt

mkdir -p "$RESULTS_DIR"

COMMON_ARGS=(
    --ckpt_path "$CKPT_PATH"
    --model_type "$MODEL_TYPE"
    --dataset "$DATASET"
    --test_task item2index
    --test_batch_size 16
    --test_prompt_ids 0
    --index_file .index_qwen7B.json
    --ratio_dataset 1
    --base_model "$BASE_MODEL"
    --num_beams 10
    --results_file "$RESULTS_FILE"
    # --lora
)

if $DEBUG; then
    python -m src.seqrec.metric "${COMMON_ARGS[@]}"
else
    nohup python -m src.seqrec.metric "${COMMON_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Metric item2index started with PID: $PID"
    echo "Logs: $LOG_FILE"
fi
