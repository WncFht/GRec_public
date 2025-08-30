export CUDA_VISIBLE_DEVICES=3

# LoRA配置
LORA_PATH=./ckpt/Instruments/Qwen2-VL-2B-Instruct-seqrec-mmitemenrich-lora-1-qwen7B/checkpoint-44948
BASE_MODEL=./ckpt/base_model/Qwen2-VL-2B-Instruct  # 基础模型路径
MODEL_TYPE=qwen2_vl

# 数据集配置
DATASET=Instruments

python -m src.text_generation.case_lora \
    --ckpt_path $LORA_PATH \
    --base_model $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --lora \
    --test_batch_size 1 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json \
    --seed 42