export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export WANDB_PROJECT=GRec
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1,2,3


# DATASET=Arts,Automotive,Cell,Games,Instruments,Pet,Tools,Toys,Sports
DATASET=Instruments

# BASE_MODEL=ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
# MODEL_TYPE=llava_onevision

BASE_MODEL=ckpt/base_model/Qwen2-VL-7B-Instruct
MODEL_TYPE=qwen2_vl

DATA_PATH=./data
RATIO_DATASET=1
OUTPUT_DIR=./ckpt/$DATASET/Qwen2-VL-7B-lora-item2index-seqrec-fusionseqrec-nonewtoken


# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Use 'tail -f $LOG_FILE' to monitor progress"


nohup torchrun --nproc_per_node=4 --master_port=33325 -m src.finetune.train_ddp_vl_nonewtoken \
    --seed 42 \
    --base_model $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --use_gradient_checkpointing \
    --num_workers 32 \
    --learning_rate 5e-5 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z2_bf16.json \
    --bf16 \
    --use_lora \
    --lora_modules_to_save "embed_tokens,lm_head" \
    --only_train_response \
    --tasks item2index,seqrec,fusionseqrec \
    --train_prompt_sample_num 1,1,1 \
    --train_data_sample_num 0,0,0 \
    --ratio_dataset $RATIO_DATASET \
    --report_to wandb \
    --index_file .index_qwen7B.json > $LOG_FILE 2>&1 &
    # --resume_from_checkpoint ckpt/Instruments/Llava-onevision-emb-item2index,seqrec-1-qwen7B/checkpoint-14984 \

# 获取进程ID
PID=$!
echo "Training started with PID: $PID"
echo "To stop training: kill $PID"

# 保存PID到文件以便后续管理
echo $PID > $OUTPUT_DIR/training.pid

tail -f $LOG_FILE
# nohup ./convert/convert.sh $OUTPUT_DIR >convert.log 2>&1 &


# /opt/meituan/dolphinfs_zhangkangning02/zkn/GRec/ckpt/Instruments/Qwen2-VL-2B-Instruct-seqrec,mmitem2index,fusionseqrec-1-qwen7B