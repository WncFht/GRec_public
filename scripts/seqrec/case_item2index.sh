export CUDA_VISIBLE_DEVICES=2

MODEL_PATH=ckpt/Instruments/Llava-onevision-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463
MODEL_TYPE=llava_onevision

# MODEL_PATH=ckpt/Instruments/LC-Rec/final-checkpoint-2446
# MODEL_TYPE=llama

# MODEL_PATH=ckpt/Instruments/Qwen2-VL-7B-Instruct-seqrec-mmitemenrich-1-qwen7B-old/checkpoint-89890
# MODEL_TYPE=qwen2_vl

python -m src.seqrec.case \
    --ckpt_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --test_task item2index \
    --test_batch_size 5 \
    --index_file .index_qwen7B.json \
    --ratio_dataset 1 \
    --num_beams 10