#!/bin/bash
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

# --- 配置文件路径 ---
CONFIG_FILE=./config/multimodal_finetune_lora.yml

echo "================================================="
echo "  Starting Multi-Modal LoRA Fine-tuning"
echo "================================================="
echo "Config File: $CONFIG_FILE"
echo "-------------------------------------------------"

# 创建日志目录
LOG_DIR=./log/multimodal_finetune
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOG_FILE="$LOG_DIR/finetune_run_${TIMESTAMP}.log"

echo "Log will be saved to: $LOG_FILE"
echo "-------------------------------------------------"

# 使用 python -m 以模块方式运行，确保相对导入正确
# 将所有输出重定向到日志文件，并在后台运行
nohup python3 -m src.multimodal_finetune_lora \
    --config_file $CONFIG_FILE > "$LOG_FILE" 2>&1 &

echo "Fine-tuning process started in the background."
echo "Check log for details: $LOG_FILE"
echo "================================================="

