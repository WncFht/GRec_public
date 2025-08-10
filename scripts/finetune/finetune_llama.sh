export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
BASE_MODEL=huggyllama/llama-7
BASE_MODEL_DIR=/hpc_stor03/sjtu_home/haotian.fang/.cache/modelscope/hub/models/skyline2006/llama-7b
MODEL_ALISE=llama-7b
DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/$MODEL_ALISE-finetune/

mkdir -p ./log
TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOG_FILE=./log/${TIMESTAMP}_${DATASET}_${MODEL_ALISE}_finetune.log

# torchrun --nproc_per_node=8 --master_port=23324 finetune.py \
nohup python3 src/multimodal_finetune.py \
    --model_type llama \
    --base_model $BASE_MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --epochs 1 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item,fusionseqrec \
    --train_prompt_sample_num 1,1,1,1 \
    --train_data_sample_num 0,0,0,0 \
    --index_file .index_llama-td.json > $LOG_FILE 2>&1 &
    # --deepspeed ./config/ds_z3_bf16.json \


# cd convert
# nohup ./convert.sh $OUTPUT_DIR >convert.log 2>&1 &
# cd ..






# DATASET=Arts
# BASE_MODEL= huggyllama/llama-7b
# DATA_PATH=./data
# OUTPUT_DIR=./ckpt/$DATASET/

# # torchrun --nproc_per_node=8 --master_port=13324 finetune.py \
# nohup python finetune.py \
#     --base_model $BASE_MODEL \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --data_path $DATA_PATH \
#     --per_device_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --epochs 4 \
#     --weight_decay 0.01 \
#     --save_and_eval_strategy epoch \
#     --deepspeed ./config/ds_z3_bf16.json \
#     --bf16 \
#     --only_train_response \
#     --tasks seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain \
#     --train_prompt_sample_num 1,1,1,1,1,1 \
#     --train_data_sample_num 0,0,0,30000,0,0 \
#     --index_file .index.json > finetune.log 2>&1 &


# cd convert
# nohup ./convert.sh $OUTPUT_DIR >convert.log 2>&1 &
# cd ..





# DATASET=Instruments
# BASE_MODEL= huggyllama/llama-7b
# DATA_PATH=./data
# OUTPUT_DIR=./ckpt/$DATASET/

# # torchrun --nproc_per_node=8 --master_port=33324 finetune.py \
# nohup python finetune.py \
#     --base_model $BASE_MODEL \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --data_path $DATA_PATH \
#     --per_device_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --epochs 4 \
#     --weight_decay 0.01 \
#     --save_and_eval_strategy epoch \
#     --deepspeed ./config/ds_z3_bf16.json \
#     --bf16 \
#     --only_train_response \
#     --tasks seqrec,item2indexexport WANDB_MODE=offline
