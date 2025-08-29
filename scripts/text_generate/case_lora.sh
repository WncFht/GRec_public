#!/bin/bash

# LoRA模型案例测试脚本
export CUDA_VISIBLE_DEVICES=0

# LoRA配置
LORA_PATH=./ckpt/Instruments/lora_text_generation/checkpoint-1000
BASE_MODEL=./ckpt/Qwen2-VL-7B-Instruct  # 基础模型路径
MODEL_TYPE=qwen2_vl

# 数据集配置
DATASET=Instruments
DATA_DIR=./data/$DATASET

python -m src.text_generation.case_lora \
    --ckpt_path $LORA_PATH \
    --base_model $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --lora \
    --test_batch_size 1 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json \
    --data_dir $DATA_DIR \
    --seed 42