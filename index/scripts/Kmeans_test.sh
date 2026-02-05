#!/bin/bash

# Limit BLAS threads to avoid OpenBLAS "too many memory regions" on high-core machines.
: "${INDEX_BLAS_NUM_THREADS:=32}"
export INDEX_BLAS_NUM_THREADS
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"

MODEL_NAME=qwen7b

# =========================
# Single vs Multi dataset
# =========================
# 单数据集（兼容旧用法）：
DATASET=Instruments
DATA_PATH=./data/$DATASET/${DATASET}.emb-${MODEL_NAME}-td.npy

# 多数据集合并训练（把下面 DATASETS 打开即可）：
# - 训练只跑一次，学习共享 codebook / encoder
# - 后续用 generate.sh 对每个数据集单独生成并落盘 index json
DATASETS=() # 例如：(Arts Games Instruments)
TRAIN_DATASET=Multi # 用于 ckpt_dir / wandb 命名（可改成 Arts+Games+...）
DATA_PATHS=()       # 可选：手动指定多个 .npy 路径；留空则按默认规则拼接

LOG_FILE="./log/index_$(date +%Y%m%d%H%M%S).log"

# --- A/B/C Test Configuration for K-Means ---
# Set KMEANS_MODE to one of the following:
# 'large': Use the new, large-scale K-Means initialization.
# 'small': Use the original, small-batch K-Means initialization.
# 'none':  Disable K-Means initialization completely.
KMEANS_MODE='none'

# --- Logic to set arguments and wandb name based on mode ---
KMEANS_INIT_ARG="false"
LARGE_SCALE_KMEANS_ARG="false"
WANDB_SUFFIX=""

if [ "$KMEANS_MODE" = "large" ]; then
  KMEANS_INIT_ARG="true"
  LARGE_SCALE_KMEANS_ARG="true"
  WANDB_SUFFIX="-LargeKMeans"
elif [ "$KMEANS_MODE" = "small" ]; then
  KMEANS_INIT_ARG="true"
  LARGE_SCALE_KMEANS_ARG="false"
  WANDB_SUFFIX="-SmallKMeans"
else
  KMEANS_INIT_ARG="false"
  LARGE_SCALE_KMEANS_ARG="false"
  WANDB_SUFFIX="-NoKMeans"
fi

WANDB_RUN_NAME="${DATASET}-${MODEL_NAME}${WANDB_SUFFIX}"
# ----------------------------------------------------

mkdir -p ./log

DATA_ARGS=()
CKPT_DIR=""
RUN_NAME=""

if [ ${#DATASETS[@]} -gt 0 ]; then
  if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    for d in "${DATASETS[@]}"; do
      DATA_PATHS+=("./data/$d/${d}.emb-${MODEL_NAME}-td.npy")
    done
  fi
  DATA_ARGS=(--data_paths "${DATA_PATHS[@]}")
  CKPT_DIR="./data/$TRAIN_DATASET/index/$MODEL_NAME/"
  RUN_NAME="${TRAIN_DATASET}-${MODEL_NAME}${WANDB_SUFFIX}"
else
  DATA_ARGS=(--data_path "$DATA_PATH")
  CKPT_DIR="./data/$DATASET/index/$MODEL_NAME/"
  RUN_NAME="${DATASET}-${MODEL_NAME}${WANDB_SUFFIX}"
fi

WANDB_RUN_NAME="$RUN_NAME"

nohup python3 -u index/main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 2048 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --dropout_prob 0.0 \
  --bn False \
  --e_dim 32 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --layers 2048 1024 512 256 128 64 \
  --kmeans_init "$KMEANS_INIT_ARG" \
  --large_scale_kmeans "$LARGE_SCALE_KMEANS_ARG" \
  --device cuda:0 \
  "${DATA_ARGS[@]}" \
  --ckpt_dir "$CKPT_DIR" \
  --use_wandb False \
  --wandb_project unifymmgrec \
  --wandb_name "$WANDB_RUN_NAME" > "$LOG_FILE" 2>&1 &
  
echo "Indexing started with K-Means Mode: $KMEANS_MODE. Log file: $LOG_FILE"
echo "W&B Run Name: $WANDB_RUN_NAME"
