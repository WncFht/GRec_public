#!/bin/bash
export WANDB_MODE=offline

# --- 配置文件路径 ---
CONFIG_FILE=./config/qwen_7B_finetune_seqrec.yml

echo "================================================="
echo "  Starting Multi-Modal LoRA Fine-tuning"
echo "================================================="
echo "Config File: $CONFIG_FILE"
echo "-------------------------------------------------"

# 创建日志目录
LOG_DIR=./log/finetune/qwen/qwen_7B
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOG_FILE="$LOG_DIR/finetune-seqrec-qwen7B-all-${TIMESTAMP}.log"

echo "Log will be saved to: $LOG_FILE"
echo "-------------------------------------------------"

# 使用 python -m 以模块方式运行，确保相对导入正确
# 将所有输出重定向到日志文件，并在后台运行
torchrun --nproc_per_node=2 --master_port=13324 -m src.finetune.multimodal_finetune \
    --config_file $CONFIG_FILE 2>&1 | tee $LOG_FILE 
    #  > "$LOG_FILE" 2>&1 &

echo "Fine-tuning process started in the background."
echo "Check log for details: $LOG_FILE"
echo "================================================="

