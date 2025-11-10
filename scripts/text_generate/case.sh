export CUDA_VISIBLE_DEVICES=1
MODEL_PATH=ckpt/Instruments/Qwen2-VL-2B-Instruct-seqrec,mmitem2index,fusionseqrec-1-qwen7B-5e-5/final-checkpoint-1835
MODEL_TYPE=qwen2_vl

python -m src.text_generation.case \
    --ckpt_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --test_batch_size 8 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json