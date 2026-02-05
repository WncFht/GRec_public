#!/bin/bash
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate grec
export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/grec/lib:$LD_LIBRARY_PATH
export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/grec/bin:$PATH
export CC=$CONDA_PREFIX/bin/conda-cc-with-crypt.sh
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export TRITON_CACHE_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/.cache/triton

export TORCH_NCCL_TRACE_BUFFER_SIZE=100000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

nvidia-smi -L

DEBUG=false
while [[ $# -gt 0 ]]; do
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

export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
export GREC_DIR=$HOME_DIR/GRec

cd $GREC_DIR

export WANDB_MODE=offline
export WANDB_DIR=$GREC_DIR

export WANDB_LOG_MODEL=false
export WANDB_ENTITY=generate_rec
export WANDB_PROJECT=minionerec
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DATASET=Instruments
CKPT_DIR=/home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/ckpt
DATA_PATH=/home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/data

OUTPUT_DIR=$CKPT_DIR/$DATASET/qwen2.5_instruct_seqrec_ranking_useprm_noformat_kl5e-3_epoch10-prefix
export WANDB_NAME=qwen2.5_instruct_seqrec_ranking_useprm_noformat_kl5e-3_epoch10-prefix
INDEX_FILE=.index_qwen3-embedding-4B.json
TASK=seqrec

BASE_MODEL=$CKPT_DIR/Instruments/Qwen2.5-3B-Instruct-sft-index_qwen3-embedding-4B-5e-5/checkpoint-12294
MODEL_TYPE=qwen2_5_instruct

CHECKPOINT_NAME=$(basename "$BASE_MODEL")
MODEL_DIR_NAME=$(basename "$(dirname "$BASE_MODEL")")
LOG_FILE="log/${MODEL_DIR_NAME}-${CHECKPOINT_NAME}-${TASK}-${TIMESTAMP}.log"
REWARD_FUNCS="format,rule,ndcg"
REWARD_WEIGHTS="0.0000001,1,1"

COMMON_ARGS=(
    --model_type "$MODEL_TYPE"
    --base_model "$BASE_MODEL"
    --train_batch_size 64
    --eval_batch_size 128
    --num_train_epochs 4
    --gradient_accumulation_steps 4
    --eval_step 0.125
    --eval_on_test
    --test_during_training
    --num_generations 16
    --beam_search
    --temperature 1.0
    --max_completion_length 128
    --learning_rate 1e-5
    --beta 5e-3
    --data_path "$DATA_PATH"
    --dataset "$DATASET"
    --index_file "$INDEX_FILE"
    --output_dir "$OUTPUT_DIR"
    --tasks "$TASK"
    --train_prompt_sample_num 1
    --train_data_sample_num 0
    --bf16
    --reward_funcs "$REWARD_FUNCS"
    --reward_weights "$REWARD_WEIGHTS"
    --noscale
    --use_prm
    --prm_match_mode prefix
)

RUN_ARGS=("${COMMON_ARGS[@]}")

if $DEBUG; then
    export CUDA_VISIBLE_DEVICES=0
    /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/python -m src.rl.rl "${RUN_ARGS[@]}"
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    mkdir -p log
    nohup accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 4 --main_process_port 29503 \
        --module src.rl.rl "${RUN_ARGS[@]}" \
        > >(tee "$LOG_FILE") 2>&1 &
    # PID=$!
    # echo "Training started with PID: $PID"
    # echo "To stop training: kill $PID"
    # echo "$LOG_FILE"
    # wait "$PID"
fi
