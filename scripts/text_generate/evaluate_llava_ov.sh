export CUDA_VISIBLE_DEVICES=1

# MODEL_PATH=ckpt/Instruments/Llava-OneVision-7B-ov-seqrec-mmitemenrich-1-qwen7B/checkpoint-179780
MODEL_PATH=ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
MODEL_TYPE=llava_onevision

python -m src.text_generation.evaluate \
    --ckpt_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --benchmark_metrics bleu,rouge,semantic_similarity,bert_score\
    --test_batch_size 4 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json