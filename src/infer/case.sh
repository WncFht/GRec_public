export CUDA_VISIBLE_DEVICES=4

MODEL_PATH=./ckpt/Instruments/Qwen2.5-VL-3B-Instruct-seqrec-item2index-1-qwen7B/checkpoint-48696
MODEL_TYPE=qwen2_5_vl
python -m src.infer.test_seqrec_with_item_id \
    --ckpt_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --test_batch_size 5 \
    --index_file .index_qwen7B.json \
    --ratio_dataset 1 \
    --num_beams 10