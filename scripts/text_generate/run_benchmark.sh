#!/bin/bash

# =====================================================================================
# 新一代文本生成与评估 Benchmark 启动脚本
#
# 该脚本是运行文本丰富 (text-enrich) 任务评测的主要入口。
# 它会协调所有步骤，从加载模型到生成文本，再到实时评估，一气呵成。
#
# 使用方法:
# 1. 配置下面的环境变量，确保它们指向正确的数据和模型路径。
# 2. (可选) 进入 config/text_generation_benchmark.yml 文件，
#    启用或禁用你想要评测的模型。
# 3. 直接运行此脚本: ./scripts/text_generate/run_benchmark.sh
# =====================================================================================

# --- 环境与路径配置 ---

# 安全设置：如果任何命令失败，则立即退出脚本
set -e

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=0

# --- 关键路径设置 ---
# 以下导出的环境变量将自动被 YAML 配置文件读取和使用

# 数据集名称和根目录
export DATASET=Instruments
export DATA_PATH=./data

# 基座模型路径 (请确保此路径真实存在)
export BASE_MODEL_NAME=Qwen2.5-VL-3B-Instruct
export BASE_MODEL_DIR=./ckpt/base_model/$BASE_MODEL_NAME

# 微调后模型的路径 (请确保此路径真实存在)
# 注意：这里通常指向包含 LoRA 权重和配置的 checkpoint 目录
export FINETUNED_MODEL_DIR=./ckpt/$DATASET/$BASE_MODEL_NAME/checkpoint-30435

# 日志目录
LOG_DIR=./log/benchmark
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/benchmark_run_$(date +%Y-%m-%d_%H-%M-%S).log


# --- 运行主评估程序 ---

CONFIG_FILE=./config/text_generation_benchmark.yml

echo "================================================="
echo "  Starting Text Generation & Evaluation Benchmark"
echo "================================================="
echo "Dataset: $DATASET"
echo "Base Model: $BASE_MODEL_DIR"
echo "Finetuned Model: $FINETUNED_MODEL_DIR"
echo "Config File: $CONFIG_FILE"
echo "Log will be saved to: $LOG_FILE"
echo "-------------------------------------------------"

# 使用 python -m 以模块方式运行，确保相对导入正确
# 将所有输出 (stdout 和 stderr) 都重定向到日志文件
python3 -m src.text_generation.main \
    --config_file $CONFIG_FILE 2>&1 | tee $LOG_FILE

echo "-------------------------------------------------"
echo "Benchmark finished!"
echo "Check the results in the directory specified in your YAML config (results_dir)."
echo "Detailed log available at: $LOG_FILE"
echo "=================================================" 