#!/bin/bash
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate grec
export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/grec/lib:$LD_LIBRARY_PATH
export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/envs/grec/bin:$PATH
export CC=$CONDA_PREFIX/bin/conda-cc-with-crypt.sh
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export TRITON_CACHE_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/.cache/triton

set -euo pipefail

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
export WANDB_DIR=$GREC_DIR/wandb
export WANDB_CACHE_DIR=$GREC_DIR/.cache/wandb
export WANDB_DATA_DIT=$GREC_DIR/.cache/wandb-data
export WANDB_ARTIFACT_DIR=$GREC_DIR/artifacts

export WANDB_LOG_MODEL=false
export WANDB_ENTITY=wncfht
export WANDB_PROJECT=GRec_rl
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DATASET=Instruments
CKPT_PATH=$HOME_DIR/ckpt
OUTPUT_DIR=$CKPT_PATH/$DATASET/llava_rl_seqrec_ranking_noscale_clip
DATA_PATH=$HOME_DIR/data

export WANDB_NAME=llava_rl_seqrec_ranking_noscale_clip
INDEX_FILE=.index_qwen7B.json
TASK=seqrec

BASE_MODEL=$CKPT_PATH/Instruments/Llava-onevision-finetune-item2index-seqrec-fusionseqrec/checkpoint-4098
MODEL_TYPE=llava_onevision

CHECKPOINT_NAME=$(basename "$BASE_MODEL")
MODEL_DIR_NAME=$(basename "$(dirname "$BASE_MODEL")")
LOG_FILE="${GREC_DIR}/log/hope/${MODEL_DIR_NAME}-${CHECKPOINT_NAME}-${TASK}-clip-${TIMESTAMP}.log"
REWARD_FUNCS="format,rule,ndcg"
REWARD_WEIGHTS="1,1,1"

# PPO-style clipping settings
CLIP_RATIO=0.2
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=3.0

COMMON_ARGS=(
    --model_type "$MODEL_TYPE"
    --base_model "$BASE_MODEL"
    --train_batch_size 64
    --eval_batch_size 128
    --num_train_epochs 1
    --gradient_accumulation_steps 2
    --eval_step 0.0999
    --reward_type ranking
    --test_during_training
    --num_generations 16
    --beam_search
    --temperature 1.0
    --max_completion_length 128
    --learning_rate 1e-5
    --beta 1e-3
    --data_path "$DATA_PATH"
    --dataset "$DATASET"
    --index_file "$INDEX_FILE"
    --output_dir "$OUTPUT_DIR"
    --tasks "$TASK"
    --train_prompt_sample_num 1
    --train_data_sample_num 0
    --bf16
    --log_completions
    --completion_log_interval 100
    --reward_funcs "$REWARD_FUNCS"
    --reward_weights "$REWARD_WEIGHTS"
    --noscale
    --clip
    --clip_ratio "$CLIP_RATIO"
    --clip_ratio_low "$CLIP_RATIO_LOW"
    --clip_ratio_high "$CLIP_RATIO_HIGH"
    --clip_ratio_c "$CLIP_RATIO_C"
)

RUN_ARGS=("${COMMON_ARGS[@]}")

if $DEBUG; then
    python -m src.rl.rl "${RUN_ARGS[@]}"
else
    mkdir -p log
    nohup accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 4 --main_process_port 29503 \
        --module src.rl.rl "${RUN_ARGS[@]}" \
        > >(tee "$LOG_FILE") 2>&1 &

    PID=$!
    echo "Training started with PID: $PID"
    echo "To stop training: kill $PID"
    echo "$LOG_FILE"
    wait "$PID"
fi
