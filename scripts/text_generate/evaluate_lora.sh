#!/bin/bash

# LoRA模型评估脚本
export CUDA_VISIBLE_DEVICES=0

# LoRA配置
LORA_PATH=./ckpt/Instruments/Qwen2-VL-2B-Instruct-seqrec-mmitemenrich-lora-1-qwen7B/checkpoint-44948
BASE_MODEL=./ckpt/base_model/Qwen2-VL-2B-Instruct  # 基础模型路径
MODEL_TYPE=qwen2_vl
MODEL_NAME=qwen2_vl_lora

# 数据集配置
DATASET=Instruments
# 评估指标
METRICS=bleu,rouge,semantic_similarity,bert_score  # 可选: bleu,rouge,semantic_similarity,bert_score

# 结果保存路径
RESULTS_FILE=./results/text_generation/${DATASET}_lora_results.csv

python -m src.text_generation.evaluate_lora \
    --ckpt_path $LORA_PATH \
    --base_model $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --lora \
    --test_batch_size 8 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json \
    --benchmark_metrics $METRICS \
    --results_file $RESULTS_FILE \
    --seed 42