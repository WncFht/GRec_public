set -e

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=6

# --- 运行主评估程序 ---

CONFIG_FILE=./config/qwen_2_vl_finetune_seqrec_with_id_1.yml

echo "================================================="
echo "  Starting Sequence Recommendation Benchmark"
echo "================================================="
echo "Config File: $CONFIG_FILE"
echo "-------------------------------------------------"

# 使用 python -m 以模块方式运行，确保相对导入正确
# 使用 tee 同时将输出打印到控制台和日志文件
LOG_DIR=./log/qwen2_vl
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/qwen-2-vl-2b-finetune-qwen7B-with-id-1-freeze-$(date +%Y-%m-%d_%H-%M-%S).log

echo "Log will be saved to: $LOG_FILE"
echo "-------------------------------------------------"

python3 -m src.finetune.qwen_2_vl_finetune_with_id_freeze \
    --config_file $CONFIG_FILE  | tee $LOG_FILE
