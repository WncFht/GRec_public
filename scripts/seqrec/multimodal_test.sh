#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
DATA_PATH=./data
MODEL_BASE=Qwen2.5-VL-3B-Instruct
BASE_MODEL_DIR=./ckpt/base_model/$MODEL_BASE
LOAD_STEP=6087  # 根据自己的实际情况加载
CKPT_PATH=./ckpt/$DATASET/$MODEL_BASE/checkpoint-$LOAD_STEP
# CKPT_PATH=./ckpt/$DATASET/$MODEL_BASE
RESULTS_FILE=./results/$DATASET/$MODEL_BASE/test.json
echo "DEBUG: BASE_MODEL_DIR set to: $BASE_MODEL_DIR"
echo "DEBUG: CKPT_PATH set to: $CKPT_PATH"

# 创建日志目录
mkdir -p ./log

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# 构建日志文件名
LOG_FILE="./log/${TIMESTAMP}_${DATASET}_${MODEL_BASE}_test.log"

echo "日志文件: $LOG_FILE"

nohup python3 src/seqrec/evaluate.py \
    --gpu_id 0 \
    --base_model $BASE_MODEL_DIR \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 8 \
    --num_beams 20 \
    --test_prompt_ids all \
    --index_file .index_qwen7B.json \
    --lora \
    --test_task seqrec \
    --use_constrained_generation > "$LOG_FILE" 2>&1 &
