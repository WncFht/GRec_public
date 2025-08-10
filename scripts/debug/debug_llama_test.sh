#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
DATA_PATH=./data
MODEL_BASE=llama-7b
MODEL_ALISE=llama-7b-lora
# BASE_MODEL_DIR=./ckpt/base_model/$MODEL_BASE
BASE_MODEL_DIR=/hpc_stor03/sjtu_home/haotian.fang/.cache/modelscope/hub/models/skyline2006/llama-7b
# LOAD_STEP=30435 # 根据自己的实际情况加载
# CKPT_PATH=./ckpt/$DATASET/$MODEL_BASE/checkpoint-$LOAD_STEP
# CKPT_PATH=./ckpt/$DATASET/$MODEL_BASE # 也可以加载整个模型，而不仅仅是适配器
# CKPT_PATH=./ckpt/$DATASET/$MODEL_BASE
# LOAD_CKPT=checkpoint-8547 
CKPT_PATH=./ckpt/$DATASET/$MODEL_ALISE

echo "--- 启动调试脚本 ---"
echo "使用模型基础: $MODEL_BASE"
echo "使用数据集: $DATASET"
echo "加载检查点: $CKPT_PATH"
echo "----------------------"

# 直接在前台运行调试脚本，以便观察输出
python3 src/debug_test.py \
    --model_type llama \
    --base_model $BASE_MODEL_DIR \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --test_batch_size 2 \
    --num_beams 5 \
    --test_prompt_ids 0 \
    --index_file .index_llama-td.json \
    --test_task seqrec \
    --lora \
    --use_constrained_generation 