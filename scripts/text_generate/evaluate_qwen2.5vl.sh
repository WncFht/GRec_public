export CUDA_VISIBLE_DEVICES=1

MODEL_PATH=./ckpt/Instruments/Qwen2.5-VL-3B-Instruct-seqrec-mmitemenrich-1-llama/checkpoint-89892
MODEL_TYPE=qwen2_5_vl

python -m src.text_generation.evaluate \
    --ckpt_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --benchmark_metrics bleu,rouge,semantic_similarity,bert_score\
    --test_batch_size 4 \
    --test_prompt_ids 0 \
    --index_file .index_llama.json