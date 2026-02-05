#!/bin/bash
set -euo pipefail

DEBUG=false
FORCE_ROLLOUT=false
SKIP_ROLLOUT=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG=true
            shift
            ;;
        --force-rollout)
            FORCE_ROLLOUT=true
            shift
            ;;
        --skip-rollout)
            SKIP_ROLLOUT=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES=0,1,2,3

TASK=seqrec
DATASET=Instruments
RATIO=1
USE_LORA=false
HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
# CKPT_PATH=ckpt/Instruments/Qwen2-VL-7B-lora-item2index-seqrec-fusionseqrec-nonewtoken/checkpoint-7284
# BASE_MODEL=./ckpt/base_model/Qwen2-VL-7B-Instruct
# MODEL_TYPE=qwen2_vl


CKPT_PATH=$HOME_DIR/ckpt/Instruments/Qwen2.5-3B-Instruct-sft-index_qwen3-embedding-4B-multi/checkpoint-11645
MODEL_TYPE=qwen2_5_instruct

DATA_PATH=$HOME_DIR/data
INDEX_FILE=.index_qwen3-embedding-4B.json
BATCH_SIZE=16
NUM_BEAMS=50
MAX_NEW_TOKENS=4
MASTER_PORT=33320
NUM_GPUS=4
EVAL_SPLIT=test
RESULTS_BASE_DIR=./results/"${EVAL_SPLIT}"

CHECKPOINT_NAME=$(basename "$CKPT_PATH")
MODEL_DIR_NAME=$(basename "$(dirname "$CKPT_PATH")")
RUN_DIR=${RESULTS_BASE_DIR}/${TASK}-constrained/${MODEL_DIR_NAME}/${CHECKPOINT_NAME}
RESULTS_FILE=${RUN_DIR}/results.json
ROLLOUT_FILE=${RUN_DIR}/rollout.json
LOG_FILE=${RUN_DIR}/log.txt

mkdir -p "$RUN_DIR"
echo "结果将保存到: $RESULTS_FILE"
echo "Rollout 将保存到/读取自: $ROLLOUT_FILE"
echo "日志将保存到: $LOG_FILE"

COMMON_ARGS=(
    --model_type "$MODEL_TYPE"
    --ckpt_path "$CKPT_PATH"
    --ratio_dataset "$RATIO"
    --dataset "$DATASET"
    --data_path "$DATA_PATH"
    --test_task "$TASK"
    --test_batch_size "$BATCH_SIZE"
    --num_beams "$NUM_BEAMS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --index_file "$INDEX_FILE"
    --test_prompt_ids "0"
    --results_file "$RESULTS_FILE"
    --rollout_file "$ROLLOUT_FILE"
    --eval_split "$EVAL_SPLIT"
    --metrics hit@1,hit@3,hit@5,hit@10,hit@20,hit@50,ndcg@1,ndcg@3,ndcg@5,ndcg@10,ndcg@20,ndcg@50
)

if $USE_LORA; then
    COMMON_ARGS+=(--lora --base_model "$BASE_MODEL")
fi

if $FORCE_ROLLOUT; then
    COMMON_ARGS+=(--force_rollout)
fi

if $SKIP_ROLLOUT; then
    COMMON_ARGS+=(--skip_rollout)
fi

if $DEBUG; then
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        -m src.seqrec.metric_constrained_ddp "${COMMON_ARGS[@]}"
else
    nohup torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        -m src.seqrec.metric_constrained_ddp "${COMMON_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Constrained DDP testing started with PID: $PID"
    echo "Logs: $LOG_FILE"
fi
