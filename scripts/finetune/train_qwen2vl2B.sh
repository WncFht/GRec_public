#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export WANDB_PROJECT=GRec
export PYTHONUNBUFFERED=1

DATASET=Instruments
BASE_MODEL=./ckpt/base_model/Qwen2-VL-2B-Instruct
MODEL_TYPE=qwen2_vl
DATA_PATH=./data
RATIO_DATASET=1
OUTPUT_DIR=./ckpt/$DATASET/Qwen2-VL-2B-Instruct-mmitemenrich-$RATIO_DATASET-qwen7B

# 检查是否为调试模式
DEBUG_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--debug" ]; then
        DEBUG_MODE=true
        break
    fi
done

# 构建基础命令
CMD="python -m src.finetune.unified_multitask_train \
    --seed 42 \
    --base_model $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 1 \
    --epochs 6 \
    --gradient_accumulation_steps 2 \
    --use_gradient_checkpointing \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --freeze visual \
    --only_train_response \
    --tasks mmitemenrich \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --ratio_dataset $RATIO_DATASET \
    --report_to wandb \
    --index_file .index_qwen7B.json"

# 根据调试模式执行
if [ "$DEBUG_MODE" = true ]; then
    echo "Running in DEBUG mode (console output)..."
    $CMD --debug
else
    echo "Running in PRODUCTION mode (file output)..."
    
    # 确保输出目录存在
    mkdir -p $OUTPUT_DIR
    
    # 生成日志文件名
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$OUTPUT_DIR/training_${TIMESTAMP}.log"
    
    echo "Logging to: $LOG_FILE"
    echo "Use 'tail -f $LOG_FILE' to monitor progress"
    
    # 使用nohup运行，输出重定向到日志文件
    nohup $CMD > $LOG_FILE 2>&1 &
    
    # 获取进程ID
    PID=$!
    echo "Training started with PID: $PID"
    echo "To stop training: kill $PID"
    
    # 保存PID到文件以便后续管理
    echo $PID > $OUTPUT_DIR/training.pid
fi