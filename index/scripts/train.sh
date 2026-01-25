#!/bin/bash

# Limit BLAS threads to avoid OpenBLAS "too many memory regions" on high-core machines.
: "${INDEX_BLAS_NUM_THREADS:=32}"
export INDEX_BLAS_NUM_THREADS
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"

# 建议在项目根目录（src/GRec）下执行：
#   bash index/scripts/train.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# =========================
# OpenOneRec tokenizer config
# =========================
# OpenOneRec 默认：n_layers=3, codebook_size=8192, dim=4096（dim 由输入数据决定）
: "${OPENONEREC_N_LAYERS:=3}"
: "${OPENONEREC_CODEBOOK_SIZE:=1024}"
: "${OPENONEREC_LAST_SK_EPSILON:=0.003}"
: "${OPENONEREC_KMEANS_ITERS:=100}"

NUM_EMB_LIST=()
SK_EPSILONS=()
for ((i = 0; i < OPENONEREC_N_LAYERS; i++)); do
  NUM_EMB_LIST+=("$OPENONEREC_CODEBOOK_SIZE")
  if [ $i -eq $((OPENONEREC_N_LAYERS - 1)) ]; then
    SK_EPSILONS+=("$OPENONEREC_LAST_SK_EPSILON")
  else
    SK_EPSILONS+=("0.0")
  fi
done

# =========================
# Data config
# =========================
# 单数据集（兼容旧用法）

: "${ROOT_DIR:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"
: "${DATASET:=Instruments}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${DATA_PATH:=${ROOT_DIR}/data/$DATASET/${DATASET}.emb-${MODEL_NAME}-td.npy}"

# 多数据集合并训练（需要时手动修改数组）
DATASETS=(Arts Automotive Cell Games Pet Sports Tools Toys Instruments)   # 例如：(Arts Games Instruments)
DATA_PATHS=() # 可选：手动指定多个 .npy 路径；留空则按默认规则拼接
TRAIN_DATASET=Multi

# =========================
# Train config
# =========================
: "${DEVICE:=cuda:0}"
: "${LR:=1e-3}"
: "${EPOCHS:=10000}"
: "${BATCH_SIZE:=4096}"
: "${WEIGHT_DECAY:=1e-4}"
: "${LR_SCHEDULER_TYPE:=linear}"
: "${DROPOUT_PROB:=0.0}"
: "${BN:=False}"
: "${E_DIM:=32}"
: "${QUANT_LOSS_WEIGHT:=1.0}"
: "${BETA:=0.25}"
: "${LAYERS:=2048 1024 512 256 128 64}"

: "${KMEANS_INIT_ARG:=true}"
: "${LARGE_SCALE_KMEANS_ARG:=true}"
: "${KMEANS_ITERS:=$OPENONEREC_KMEANS_ITERS}"

: "${USE_WANDB:=False}"
: "${WANDB_PROJECT:=unifymmgrec}"

mkdir -p ./log
LOG_FILE="${LOG_FILE:-./log/index_train_$(date +%Y%m%d%H%M%S).log}"

DATA_ARGS=()
CKPT_ROOT=""
RUN_NAME=""

if [ ${#DATASETS[@]} -gt 0 ]; then
  if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    for d in "${DATASETS[@]}"; do
      DATA_PATHS+=("${ROOT_DIR}/data/$d/${d}.emb-${MODEL_NAME}-td.npy")
    done
  fi
  DATA_ARGS=(--data_paths "${DATA_PATHS[@]}")
  CKPT_ROOT="./data/$TRAIN_DATASET/index/$MODEL_NAME/"
  RUN_NAME="${TRAIN_DATASET}-${MODEL_NAME}"
else
  DATA_ARGS=(--data_path "$DATA_PATH")
  CKPT_ROOT="./data/$DATASET/index/$MODEL_NAME/"
  RUN_NAME="${DATASET}-${MODEL_NAME}"
fi

VQ_TAG="$(IFS=-; echo "${NUM_EMB_LIST[*]}")"
CKPT_TAG_DEFAULT="rq${#NUM_EMB_LIST[@]}_cb${VQ_TAG}"
: "${CKPT_TAG:=$CKPT_TAG_DEFAULT}"

: "${CKPT_DIR:=${CKPT_ROOT}${CKPT_TAG}/}"
RUN_NAME="${RUN_NAME}-${CKPT_TAG}"

WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"

nohup python3 -u index/main.py \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --dropout_prob "$DROPOUT_PROB" \
  --bn "$BN" \
  --e_dim "$E_DIM" \
  --quant_loss_weight "$QUANT_LOSS_WEIGHT" \
  --beta "$BETA" \
  --num_emb_list "${NUM_EMB_LIST[@]}" \
  --sk_epsilons "${SK_EPSILONS[@]}" \
  --layers $LAYERS \
  --kmeans_init "$KMEANS_INIT_ARG" \
  --large_scale_kmeans "$LARGE_SCALE_KMEANS_ARG" \
  --kmeans_iters "$KMEANS_ITERS" \
  --device "$DEVICE" \
  "${DATA_ARGS[@]}" \
  --ckpt_dir "$CKPT_DIR" \
  --use_wandb "$USE_WANDB" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$WANDB_RUN_NAME" > >(tee "$LOG_FILE") 2>&1 &

echo "Index training started. Log file: $LOG_FILE"
echo "W&B Run Name: $WANDB_RUN_NAME"
