#!/bin/bash
export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
BASE_MODEL_DIR=/hpc_stor03/sjtu_home/haotian.fang/.cache/modelscope/hub/models/skyline2006/llama-7b
MODEL_ALISE=llama-7b
DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/$MODEL_ALISE-lora/

# 创建日志目录
mkdir -p ./log
mkdir -p $OUTPUT_DIR

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# 构建日志文件名
LOG_FILE=./log/${TIMESTAMP}_${DATASET}_${MODEL_ALISE}_lora.log

echo "日志文件: $LOG_FILE"

nohup python3 src/multimodal_finetune_lora.py \
    --model_type llama \
    --base_model $BASE_MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --epochs 1 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item,fusionseqrec \
    --train_prompt_sample_num 1,1,1,1 \
    --train_data_sample_num 0,0,0,0 \
    --index_file .index_llama-td.json \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj > "$LOG_FILE" 2>&1 & 