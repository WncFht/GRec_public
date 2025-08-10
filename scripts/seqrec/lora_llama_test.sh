#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# --- 配置 ---
DATASET=Instruments
DATA_PATH=./data

# LLaMA 模型配置 (与 lora_llama.sh 对应)
MODEL_ALISE=llama-7b-lora
BASE_MODEL_DIR=/hpc_stor03/sjtu_home/haotian.fang/.cache/modelscope/hub/models/skyline2006/llama-7b

# LoRA Checkpoint 路径
# 注意: 请将 LOAD_STEP 修改为您需要测试的实际 checkpoint 步数或epoch数
# 例如: checkpoint-1000 或 epoch_3
LOAD_CKPT=epoch_3 
CKPT_PATH=./ckpt/$DATASET/$MODEL_ALISE/$LOAD_CKPT

# 结果文件路径
RESULTS_FILE=./results/$DATASET/$MODEL_ALISE/test_on_${LOAD_CKPT}.json

echo "--- LLaMA LoRA Test ---"
echo "Base Model: $BASE_MODEL_DIR"
echo "LoRA Ckpt: $CKPT_PATH"
echo "Results will be saved to: $RESULTS_FILE"
echo "----------------------"

# 创建日志和结果目录
mkdir -p ./log
mkdir -p "$(dirname "$RESULTS_FILE")"

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# 构建日志文件名
LOG_FILE="./log/${TIMESTAMP}_${DATASET}_${MODEL_ALISE}_test.log"

echo "日志文件: $LOG_FILE"

nohup python3 src/seqrec/evaluate.py \
    --model_type llama \
    --gpu_id 0 \
    --base_model "$BASE_MODEL_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --results_file "$RESULTS_FILE" \
    --test_batch_size 8 \
    --num_beams 20 \
    --test_prompt_ids all \
    --index_file .index_llama-td.json \
    --lora \
    --test_task seqrec \
    --use_constrained_generation > "$LOG_FILE" 2>&1 &

echo "LLaMA test started. Log will be written to $LOG_FILE" 