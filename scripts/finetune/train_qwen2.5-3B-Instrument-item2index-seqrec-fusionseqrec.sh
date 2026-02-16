source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate grec
export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/grec/lib:$LD_LIBRARY_PATH
export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/grec/bin:$PATH
export CC=$CONDA_PREFIX/bin/conda-cc-with-crypt.sh
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export TRITON_CACHE_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/.cache/triton
nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES

export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
export GREC_DIR=$HOME_DIR/GRec

cd $GREC_DIR
export WANDB_DIR=$GREC_DIR

DEBUG=false
while [ $# -gt 0 ]; do
    case "$1" in
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export WANDB_ENTITY=generate_rec
export WANDB_PROJECT=grec_sft
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1,2,3


# DATASET=Arts,Automotive,Cell,Games,Instruments,Pet,Tools,Toys,Sports
DATASET=Instruments

# BASE_MODEL=ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
# MODEL_TYPE=llava_onevision

BASE_MODEL=$HOME_DIR/ckpt/base_model/Qwen2.5-3B-Instruct
MODEL_TYPE=qwen2_5_instruct

DATA_PATH=$HOME_DIR/data
OUTPUT_DIR=$HOME_DIR/ckpt/$DATASET/Qwen2.5-3B-Instruct-sft-index_qwen3-embedding-4B-multi


# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Use 'tail -f $LOG_FILE' to monitor progress"

COMMON_ARGS=(
    --seed 42
    --base_model "$BASE_MODEL"
    --model_type "$MODEL_TYPE"
    --output_dir "$OUTPUT_DIR"
    --dataset "$DATASET"
    --data_path "$DATA_PATH"
    --per_device_batch_size 8
    --gradient_accumulation_steps 4
    --use_gradient_checkpointing
    --num_workers 32
    --lr_scheduler_type cosine
    --learning_rate 5e-5
    --epochs 10
    --save_and_eval_strategy steps
    --save_and_eval_steps 0.01
    --weight_decay 0.01
    --deepspeed ./config/ds_z2_bf16.json
    --bf16
    --only_train_response
    --tasks item2index,seqrec,fusionseqrec
    --train_prompt_sample_num 1,1,1
    --train_data_sample_num 0,0,0
    --ratio_dataset 1
    --report_to wandb
    --index_file .index_qwen3-embedding-4B.json
)
    # --eval_by_dataset
    # --eval_main_dataset Instruments

if $DEBUG; then
    echo "Running in DEBUG mode (single GPU, foreground)..."
    export CUDA_VISIBLE_DEVICES=0

    python -m src.finetune.train_ddp \
        "${COMMON_ARGS[@]}" \
        --debug
else
    nohup torchrun --nproc_per_node=4 --master_port=33326 -m src.finetune.train_ddp \
        "${COMMON_ARGS[@]}" > "$LOG_FILE" 2>&1 &
        # --use_lora \
        # --lora_modules_to_save "embed_tokens,lm_head" \
        # --resume_from_checkpoint ckpt/Instruments/Llava-onevision-emb-item2index,seqrec-1-qwen7B/checkpoint-14984 \

    # 获取进程ID
    PID=$!
    echo "Training started with PID: $PID"
    echo "To stop training: kill $PID"

    echo $PID > $OUTPUT_DIR/training.pid

    tail -f $LOG_FILE
fi
