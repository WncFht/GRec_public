#!/bin/bash
export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="unifymmgrec_t5_test"

# --- 配置 ---
DATASET=Instruments
DATA_PATH=./data

# T5 基础模型的名称或路径
MODEL_BASE=t5-base 
# BASE_MODEL_DIR=./ckpt/base_model/$MODEL_BASE
BASE_MODEL_DIR=/hpc_stor03/sjtu_home/haotian.fang/src/MQL4GRec/log/Instruments/ckpt_b1024_lr1e-3_seqrec/pretrain
# 根据你训练时保存的检查点步骤来设置
LOAD_STEP=2520
# CKPT_PATH=./ckpt/$DATASET/$MODEL_BASE/checkpoint-$LOAD_STEP
CKPT_PATH=$BASE_MODEL_DIR/checkpoint-$LOAD_STEP
# --- 脚本逻辑 ---
WANDB_RUN_NAME="${DATASET}_${MODEL_BASE}_test"

# 创建结果目录
RESULTS_DIR=./results/$DATASET/$MODEL_BASE
mkdir -p $RESULTS_DIR
RESULTS_FILE=$RESULTS_DIR/test_results.json

# 创建日志目录
LOG_DIR=./log/$DATASET/${MODEL_BASE}_test_logs
mkdir -p $LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${TIMESTAMP}_test.log"

echo "结果将保存到: $RESULTS_FILE"
echo "日志将写入: $LOG_FILE"

nohup python3 ./src/multimodal_test.py \
    --model_type t5 \
    --base_model $BASE_MODEL_DIR \
    --ckpt_path $CKPT_PATH \
    --data_path $DATA_PATH \
    --dataset $DATASET \
    --results_file $RESULTS_FILE \
    --test_batch_size 16 \
    --num_beams 20 \
    --test_prompt_ids "0" \
    --test_task "SeqRec" \
    --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
    --filter_items \
    --index_file .index_llama-td.json \
    --max_new_tokens 10 > "$LOG_FILE" 2>&1 &

echo "测试脚本已在后台启动。 PID: $!" 