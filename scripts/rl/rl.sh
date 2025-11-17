#!/bin/bash

# RL training launcher for GRec using GRec/src/rl/rl.py
# Adjust paths and accelerate config as needed.

export NCCL_IB_DISABLE=1

DATA_PATH="./data"
DATASET="Instruments"
MODEL_PATH="/path/to/your/llm"
OUTPUT_DIR="./output_rl"

accelerate launch --config_file ./config/zero2_opt.yaml \
    --num_processes 4 --main_process_port 29503 \
    ../src/rl/rl.py \
    --base_model ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --data_path ${DATA_PATH} \
    --dataset ${DATASET} \
    --task seqrec \
    --per_device_batch_size 8 \
    --epochs 1 \
    --learning_rate 1e-5 \
    --bf16
