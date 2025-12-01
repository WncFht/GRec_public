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

TASK=seqrec
DATASET=Instruments
RATIO=1

# CKPT_PATH=ckpt/Instruments/Qwen2-VL-7B-lora-item2index-seqrec-fusionseqrec-nonewtoken/checkpoint-7284
# BASE_MODEL=./ckpt/base_model/Qwen2-VL-7B-Instruct
# MODEL_TYPE=qwen2_vl

CKPT_PATH=ckpt/Instruments/Llava-onevision-lora-item2index-index2item-seqrec-fusionseqrec-1-qwen7B-1024/checkpoint-855
BASE_MODEL=./ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
MODEL_TYPE=llava_onevision

DATA_PATH=./data
INDEX_FILE=.index_qwen7B.json
RESULTS_DIR=./results
BATCH_SIZE=16
NUM_BEAMS=10
MAX_NEW_TOKENS=4

CHECKPOINT_NAME=$(basename "$CKPT_PATH")
MODEL_DIR_NAME=$(basename "$(dirname "$CKPT_PATH")")
RESULTS_FILE=${RESULTS_DIR}/${TASK}-constrained-${MODEL_DIR_NAME}-${CHECKPOINT_NAME}.txt
LOG_FILE=${RESULTS_FILE%.txt}_log.txt

mkdir -p "$RESULTS_DIR"
echo "结果将保存到: $RESULTS_FILE"
echo "日志将保存到: $LOG_FILE"

COMMON_ARGS=(
    --model_type "$MODEL_TYPE"
    --ckpt_path "$CKPT_PATH"
    --ratio_dataset "$RATIO"
    --dataset "$DATASET"
    --data_path "$DATA_PATH"
    --test_task "$TASK"
    --test_batch_size "$BATCH_SIZE"
    --num_beams "$NUM_BEAMS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --index_file "$INDEX_FILE"
    --test_prompt_ids "0"
    --base_model "$BASE_MODEL"
    --lora
    --results_file "$RESULTS_FILE"
)

if $DEBUG; then
    python -m src.seqrec.metric_constrained "${COMMON_ARGS[@]}"
else
    nohup python -m src.seqrec.metric_constrained "${COMMON_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Constrained testing started with PID: $PID"
    echo "Logs: $LOG_FILE"
fi
