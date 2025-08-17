export CUDA_VISIBLE_DEVICES=4

DATASET=Instruments
BASE_MODEL=Qwen/Qwen2-VL-7B-Instruct
MODEL_TYPE=qwen2_vl
DATA_PATH=./data
RATIO_DATASET=1
OUTPUT_DIR=./ckpt/$DATASET/Qwen2-VL-7B-Instruct-seqrec-$RATIO_DATASET-qwen7B

python -m src.finetune.multitask_finetune \
    --seed 42 \
    --base_model $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 2 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --freeze visual \
    --only_train_response \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --ratio_dataset $RATIO_DATASET \
    --index_file .index_qwen7B.json
