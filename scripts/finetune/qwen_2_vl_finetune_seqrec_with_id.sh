set -e

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=3

# --- 运行主评估程序 ---

CONFIG_FILE=./config/qwen_2_vl_finetune_seqrec_with_id.yml

echo "================================================="
echo "  Starting Sequence Recommendation Benchmark"
echo "================================================="
echo "Config File: $CONFIG_FILE"
echo "-------------------------------------------------"

# 使用 python -m 以模块方式运行，确保相对导入正确
# 使用 tee 同时将输出打印到控制台和日志文件
LOG_DIR=./log/seqrec_benchmark
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/qwen-2-vl-2b-finetune-qwen7B-with-id-0.01-$(date +%Y-%m-%d_%H-%M-%S).log

echo "Log will be saved to: $LOG_FILE"
echo "-------------------------------------------------"

python3 -m src.finetune.qwen_2_vl_finetune_with_id \
    --config_file $CONFIG_FILE  | tee $LOG_FILE
