#!/bin/bash

# =====================================================================================
# 新一代序列推荐评估 Benchmark 启动脚本
#
# 该脚本是运行序列推荐 (seqrec) 任务评测的统一入口。
# 所有实验配置均通过 YAML 文件管理。
#
# 使用方法:
# 1. 打开 config/seqrec_benchmark.yml 文件。
# 2. 修改或添加模型配置，并通过 `enabled: true/false` 来控制是否运行。
# 3. 直接运行此脚本: ./scripts/evaluate/run_seqrec_benchmark.sh
# =====================================================================================

# 安全设置：如果任何命令失败，则立即退出脚本
set -e

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=1

# --- 运行主评估程序 ---

CONFIG_FILE=./config/qwen_vl_finetune_seqrec_freeze_vision_test.yml

echo "================================================="
echo "  Starting Sequence Recommendation Benchmark"
echo "================================================="
echo "Config File: $CONFIG_FILE"
echo "-------------------------------------------------"

# 使用 python -m 以模块方式运行，确保相对导入正确
# 使用 tee 同时将输出打印到控制台和日志文件
LOG_DIR=./log/seqrec_benchmark
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/debug-qwen-vl-3b-finetune-qwen7B-freeze-vision-3184-$(date +%Y-%m-%d_%H-%M-%S).log

echo "Log will be saved to: $LOG_FILE"
echo "-------------------------------------------------"

python3 -m src.seqrec.evaluate \
    --config_file $CONFIG_FILE 2>&1 | tee $LOG_FILE

echo "================================================="
echo "Benchmark finished!"
echo "Check results in the 'output_dir' specified in your YAML config."
echo "Detailed log available at: $LOG_FILE"
echo "=================================================" 