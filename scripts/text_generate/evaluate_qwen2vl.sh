export CUDA_VISIBLE_DEVICES=1

# MODEL_PATH=./ckpt/base_model/Qwen2.5-VL-3B-Instruct
# MODEL_TYPE=qwen2_5_vl

MODEL_PATH=ckpt/base_model/Qwen2-VL-2B-Instruct
MODEL_TYPE=qwen2_vl

python -m src.text_generation.evaluate \
    --ckpt_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --benchmark_metrics bleu,rouge,semantic_similarity,bert_score\
    --test_batch_size 16 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json