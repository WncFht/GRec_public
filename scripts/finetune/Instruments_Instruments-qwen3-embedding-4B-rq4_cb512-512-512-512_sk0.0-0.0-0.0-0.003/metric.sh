#!/bin/bash
set -eo pipefail

DEBUG=false
FORCE_ROLLOUT=false
SKIP_ROLLOUT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) DEBUG=true; shift ;;
    --force-rollout) FORCE_ROLLOUT=true; shift ;;
    --skip-rollout) SKIP_ROLLOUT=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

DEFAULT_GREC_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec"
: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

cd "$GREC_ROOT" || exit 1

INDEX_TAG="rq4_cb512-512-512-512_sk0.0-0.0-0.0-0.003"
INDEX_MATCH_TAG="${INDEX_TAG%%_sk*}"
: "${TASK:=seqrec}"
: "${DATASET:=Instruments}"
: "${RATIO:=1}"
: "${MODEL_TYPE:=qwen2_5_instruct}"
: "${DATA_PATH:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data}"
: "${ROOT_DIR:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"
: "${INDEX_EMB_MODEL:=qwen3-embedding-4B}"
: "${INDEX_DATASETS:=Instruments}"

INDEX_DATASETS_TAG="${INDEX_DATASETS// /}"
INDEX_DATASETS_TAG="${INDEX_DATASETS_TAG//,/-}"
: "${NUM_GPUS:=4}"
: "${MASTER_PORT:=33320}"
: "${BATCH_SIZE:=16}"
: "${NUM_BEAMS:=50}"
: "${MAX_NEW_TOKENS:=4}"
: "${EVAL_SPLIT:=test}"
: "${RESULTS_BASE_DIR:=./results/$EVAL_SPLIT}"

if [[ -z "${INDEX_FILE:-}" ]]; then
  pattern="$DATA_PATH/$DATASET/${DATASET}.index_emb-${INDEX_EMB_MODEL}_${INDEX_MATCH_TAG}_ds${INDEX_DATASETS_TAG}_rid*.json"
  latest_index="$(ls -1t $pattern 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_index" ]]; then
    echo "Error: cannot find index file for tag $INDEX_TAG under $DATA_PATH/$DATASET" >&2
    echo "Hint: run index generate first or set INDEX_FILE manually." >&2
    exit 1
  fi
  base_name="$(basename "$latest_index")"
  INDEX_FILE=".${base_name#${DATASET}.}"
fi

if [[ -z "${CKPT_PATH:-}" ]]; then
  INDEX_KEY="${INDEX_FILE#.}"
  INDEX_KEY="${INDEX_KEY%.json}"
  sft_dir="${SFT_DIR:-${ROOT_DIR}/ckpt/$DATASET/qwen2.5-3b-sft__idx-$INDEX_KEY}"
  CKPT_PATH="$(ls -1dt "$sft_dir"/checkpoint-* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$CKPT_PATH" ]]; then
    echo "Error: cannot resolve CKPT_PATH from $sft_dir" >&2
    echo "Hint: set CKPT_PATH manually." >&2
    exit 1
  fi
fi

CHECKPOINT_NAME="$(basename "$CKPT_PATH")"
MODEL_DIR_NAME="$(basename "$(dirname "$CKPT_PATH")")"
RUN_DIR="$RESULTS_BASE_DIR/${TASK}-constrained/${DATASET}-${INDEX_TAG}/$MODEL_DIR_NAME/$CHECKPOINT_NAME"
RESULTS_FILE="${RESULTS_FILE:-$RUN_DIR/results.json}"
ROLLOUT_FILE="${ROLLOUT_FILE:-$RUN_DIR/rollout.json}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/log.txt}"

mkdir -p "$RUN_DIR"

echo "[bundle/metric] CKPT_PATH=$CKPT_PATH"
echo "[bundle/metric] INDEX_FILE=$INDEX_FILE"
echo "[bundle/metric] INDEX_EMB_MODEL=$INDEX_EMB_MODEL INDEX_DATASETS=$INDEX_DATASETS"
echo "[bundle/metric] RESULTS_FILE=$RESULTS_FILE"
echo "[bundle/metric] ROLLOUT_FILE=$ROLLOUT_FILE"

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
  --test_prompt_ids "${TEST_PROMPT_IDS:-0}"
  --results_file "$RESULTS_FILE"
  --rollout_file "$ROLLOUT_FILE"
  --eval_split "$EVAL_SPLIT"
  --metrics "${METRICS:-hit@1,hit@3,hit@5,hit@10,hit@20,hit@50,ndcg@1,ndcg@3,ndcg@5,ndcg@10,ndcg@20,ndcg@50}"
)

if $FORCE_ROLLOUT; then
  COMMON_ARGS+=(--force_rollout)
fi
if $SKIP_ROLLOUT; then
  COMMON_ARGS+=(--skip_rollout)
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if $DEBUG; then
  torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT"     -m src.seqrec.metric_constrained_ddp "${COMMON_ARGS[@]}"
else
  nohup torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT"     -m src.seqrec.metric_constrained_ddp "${COMMON_ARGS[@]}" > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "Constrained DDP testing started with PID: $PID"
  echo "Logs: $LOG_FILE"
fi
