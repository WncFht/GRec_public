export CUDA_VISIBLE_DEVICES=1

MODEL_PATH=ckpt/Instruments/Llava-onevision-finetuned-item2index,seqrec-1-qwen7B/checkpoint-3748
MODEL_TYPE=llava_onevision

# MODEL_PATH=ckpt/Sports/Llama-emb-item2index,index2item,seqrec-datasets-1-llama/checkpoint-31848
# MODEL_TYPE=llama

python -m src.seqrec.metric \
    --ckpt_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --test_task seqrec \
    --test_batch_size 16 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json \
    --ratio_dataset 1 \
    --num_beams 10