#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# --- T5 配置 ---
DATASET=Instruments
DATA_PATH=./data
MODEL_BASE=t5-base
BASE_MODEL_DIR=/hpc_stor03/sjtu_home/haotian.fang/src/MQL4GRec/log/Instruments/ckpt_b1024_lr1e-3_seqrec/pretrain
LOAD_STEP=1260
# CKPT_PATH=./ckpt/$DATASET/$MODEL_BASE/checkpoint-$LOAD_STEP

echo "--- 启动 T5 调试脚本 ---"
echo "使用模型基础: $MODEL_BASE"
echo "使用数据集: $DATASET"
echo "加载检查点: $CKPT_PATH"
echo "----------------------"

# 直接在前台运行调试脚本，以便观察输出
python3 src/debug_test.py \
    --model_type t5 \
    --gpu_id 0 \
    --base_model $BASE_MODEL_DIR \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --test_batch_size 2 \
    --num_beams 5 \
    --test_prompt_ids 0 \
    --index_file .index_llama-td.json \
    --test_task seqrec 