export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
CKPT_PATH=./ckpt/Instruments/Qwen2-VL-2B-Instruct-seqrec-lora-0.01-qwen7B
BASE_MODEL=./ckpt/base_model/Qwen2-VL-2B-Instruct
MODEL_TYPE=qwen2_vl
DATA_PATH=./data
RESULTS_FILE=./results/${DATASET}_lora_results.json

python -m src.seqrec.metric_seqrec_lora \
    --model_type $MODEL_TYPE \
    --ckpt_path $CKPT_PATH \
    --base_model $BASE_MODEL \
    --lora \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --test_batch_size 8 \
    --num_beams 20 \
    --index_file .index_qwen7B.json \
    --test_prompt_ids "0" \
    --results_file $RESULTS_FILE