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

SCRIPT_NAME="case_item2index"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="log"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}-${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES=2

MODEL_PATH=ckpt/Instruments/Llava-onevision-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463
MODEL_TYPE=llava_onevision

# MODEL_PATH=ckpt/Instruments/LC-Rec/final-checkpoint-2446
# MODEL_TYPE=llama

# MODEL_PATH=ckpt/Instruments/Qwen2-VL-7B-Instruct-seqrec-mmitemenrich-1-qwen7B-old/checkpoint-89890
# MODEL_TYPE=qwen2_vl

COMMON_ARGS=(
    --ckpt_path "$MODEL_PATH"
    --model_type "$MODEL_TYPE"
    --test_task item2index
    --test_batch_size 5
    --index_file .index_qwen7B.json
    --ratio_dataset 1
    --num_beams 10
)

if $DEBUG; then
    python -m src.seqrec.case "${COMMON_ARGS[@]}"
else
    mkdir -p "$LOG_DIR"
    nohup python -m src.seqrec.case "${COMMON_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Case (item2index) started with PID: $PID"
    echo "Logs: $LOG_FILE"
fi
