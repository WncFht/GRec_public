export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=7

DATASET=Instruments
BASE_MODEL= Qwen/Qwen2.5-VL-3B-Instruct
DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/

python -m src.finetune.multitask_finetune \
    --seed 42 \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 16 \
    --epochs 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item \
    --train_prompt_sample_num 1,1,1 \
    --train_data_sample_num 0,0,0 \
    --ratio_dataset 0.1 \
    --index_file .index_qwen7B.json
