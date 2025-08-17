export CUDA_VISIBLE_DEVICES=4
MODEL_PATH=./ckpt/Instruments/Qwen2-VL-7B-Instruct-seqrec-item2index-1-qwen7B/checkpoint-84892
MODEL_TYPE=qwen2_vl

python -m src.infer.test_with_id \
    --ckpt_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --test_batch_size 8 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json \
    --ratio_dataset 1 \
    --num_beams 10