#!/bin/bash
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
BASE_MODEL=Qwen2.5-VL-3B-Instruct
BASE_MODEL_DIR=./ckpt/base_model/$BASE_MODEL # 保持与LoRA脚本一致的目录结构

DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/$BASE_MODEL-finetune

# 创建日志目录
mkdir -p ./log

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# 构建日志文件名
LOG_FILE="./log/${TIMESTAMP}_${DATASET}_${BASE_MODEL}_finetune.log"

echo "日志文件: $LOG_FILE"

nohup python3 src/multimodal_finetune.py \
    --base_model $BASE_MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --epochs 1 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --only_train_response \
    --tasks seqrec,mmitem2index,mmindex2item,mmitemenrich \
    --train_prompt_sample_num 1,1,1,1 \
    --train_data_sample_num 0,0,0,0 \
    --index_file .index_qwen3b.json > "$LOG_FILE" 2>&1 &

    # --deepspeed ./config/ds_z3_bf16.json \
    # --tasks mmitem2index \
    # --train_prompt_sample_num 1 \
    # --train_data_sample_num 0 \