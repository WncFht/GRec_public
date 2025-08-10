export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

DATASET=Instruments
BASE_MODEL=Qwen/Qwen3-1.7B
DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/

# torchrun --nproc_per_node=1 --master_port=23324 src/qwen3_finetune.py \
python src/qwen3_finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item \
    --train_prompt_sample_num 1,1,1 \
    --train_data_sample_num 0,0,0 \
    --index_file .index_qwen3b.json

    # --deepspeed ./config/ds_z3_bf16.json \
    # --tasks seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain \
    # --train_prompt_sample_num 1,1,1,1,1,1 \
    # --train_data_sample_num 0,0,0,100000,0,0 \