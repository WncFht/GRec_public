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

SCRIPT_NAME="case_seqrec"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="log"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}-${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES=1

DATASET=Instruments
RATIO=1

CKPT_PATH=/opt/meituan/dolphinfs_zhangkangning02/zkn/verl/checkpoints/grec_verl/qwen2-vl-7b-n32/global_step_105/actor/merged
BASE_MODEL=./ckpt/base_model/Qwen2-VL-7B-Instruct
MODEL_TYPE=qwen2_vl

# CKPT_PATH=./ckpt/Instruments/Llava-onevision-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463
# BASE_MODEL=./ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
# MODEL_TYPE=llava_onevision

DATA_PATH=./data

COMMON_ARGS=(
    --model_type "$MODEL_TYPE"
    --ckpt_path "$CKPT_PATH"
    --base_model "$BASE_MODEL"
    --dataset "$DATASET"
    --data_path "$DATA_PATH"
    --test_batch_size 1
    --num_beams 10
    --ratio_dataset "$RATIO"
    --index_file .index_qwen7B.json
    # --lora
)

if $DEBUG; then
    python -m src.seqrec.case "${COMMON_ARGS[@]}"
else
    mkdir -p "$LOG_DIR"
    nohup python -m src.seqrec.case "${COMMON_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Case (seqrec) started with PID: $PID"
    echo "Logs: $LOG_FILE"
fi
