#!/bin/bash

# 这个脚本演示了如何使用LoRA微调，同时冻结原始词汇表的embeddings
# 只训练新增token的embeddings

export CUDA_VISIBLE_DEVICES=4

DATASET=Instruments
BASE_MODEL=Qwen/Qwen2-VL-7B-Instruct
MODEL_TYPE=qwen2_vl
DATA_PATH=./data
RATIO_DATASET=1
OUTPUT_DIR=./ckpt/$DATASET/Qwen2-VL-7B-Instruct-seqrec-lora-freeze-emb-$RATIO_DATASET-qwen7B

python -m src.finetune.multitask_lora_finetune \
    --seed 42 \
    --base_model $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 4 \
    --epochs 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --freeze embeddings \
    --only_train_response \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --ratio_dataset $RATIO_DATASET \
    --index_file .index_qwen7B.json \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_modules_to_save "embed_tokens,lm_head" \
    --use_gradient_checkpointing