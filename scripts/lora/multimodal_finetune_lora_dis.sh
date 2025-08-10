#!/bin/bash
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据你的GPU数量调整

# 训练参数
DATASET=Mini
BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
DATA_PATH=./data
DATE=July2
OUTPUT_DIR=./ckpt/$DATASET/$BASE_MODEL/

# 分布式参数
NPROC_PER_NODE=4  # GPU数量
MASTER_PORT=29500

# 启动分布式训练
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    src/multimodal_finetune_lora_dis.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --epochs 1 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --only_train_response \
    --tasks seqrec,mmitem2index,mmindex2item,mmitemenrich \
    --train_prompt_sample_num 1,1,1,1 \
    --train_data_sample_num 0,0,0,0 \
    --index_file .index_qwen7B.json \
    --lora_r 16 \
    --lora_alpha 64 \
    --lora_dropout 0.15 \
    --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
    --use_deepspeed \
    --deepspeed_config_file ./config/ds_z2_bf16.json \
    --gradient_checkpointing \
    --dataloader_num_workers 2


        # --use_deepspeed \
        #--deepspeed_config_file ./config/ds_z2_bf16.json \
