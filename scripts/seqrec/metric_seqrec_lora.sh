export CUDA_VISIBLE_DEVICES=1

DATASET=Instruments
# DATASET=Arts,Games,Instruments
RATIO=1

CKPT_PATH=ckpt/Instruments/Qwen2-VL-7B-lora-item2index-seqrec-fusionseqrec-nonewtoken/checkpoint-7284
BASE_MODEL=./ckpt/base_model/Qwen2-VL-7B-Instruct
MODEL_TYPE=qwen2_vl

# CKPT_PATH=ckpt/Instruments/Llava-onevision-finetune-item2index-index2item-seqrec-fusionseqrec-1-qwen7B/checkpoint-5619
# BASE_MODEL=./ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
# MODEL_TYPE=llava_onevision

# CKPT_PATH=ckpt/Instruments/Qwen2.5-7B-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463
# BASE_MODEL=ckpt/base_model/Qwen2.5-7B
# MODEL_TYPE=qwen

# CKPT_PATH=ckpt/Instruments/Qwen2-VL-7B-lora-item2index-seqrec-fusionseqrec-1-qwen7B-merged-old
# BASE_MODEL=ckpt/base_model/Qwen2-VL-7B-Instruct
# MODEL_TYPE=qwen2_vl

DATA_PATH=./data
RESULTS_FILE=./results/${DATASET}_lora_results.json

python -m src.seqrec.metric \
    --model_type $MODEL_TYPE \
    --ckpt_path $CKPT_PATH \
    --ratio_dataset $RATIO \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --test_batch_size 16 \
    --num_beams 10 \
    --index_file .index_qwen7B.json \
    --test_prompt_ids "0" \
    --base_model $BASE_MODEL \
    --lora \
    --results_file $RESULTS_FILE