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
: "${OPENONEREC_N_LAYERS:=4}"
: "${OPENONEREC_CODEBOOK_SIZE:=512}"
: "${OPENONEREC_LAST_SK_EPSILON:=0.003}"
: "${OPENONEREC_KMEANS_ITERS:=100}"

NUM_EMB_LIST=()
SK_EPSILONS=()
for ((i = 0; i < OPENONEREC_N_LAYERS; i++)); do
  NUM_EMB_LIST+=("$OPENONEREC_CODEBOOK_SIZE")
  if [ $i -ge 0 ]; then
    SK_EPSILONS+=("$OPENONEREC_LAST_SK_EPSILON")
  else
    SK_EPSILONS+=("0.0")
  fi
done

echo $SK_EPSILONS

# =========================
# Data config
# =========================
# 单数据集（兼容旧用法）

: "${ROOT_DIR:=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian}"
: "${DATASET:=Instruments}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${DATA_PATH:=${ROOT_DIR}/data/$DATASET/${DATASET}.emb-${MODEL_NAME}-td.npy}"

# 多数据集合并训练（需要时手动修改数组）
DATASETS=(Instruments)
# DATASETS=(Arts Automotive Cell Games Pet Sports Tools Toys Instruments)   # 例如：(Arts Games Instruments)
# DATASETS=(Games)
DATA_PATHS=() # 可选：手动指定多个 .npy 路径；留空则按默认规则拼接
TRAIN_DATASET=$(IFS=-; echo "${DATASETS[*]}")

# =========================
# Train config
# =========================
: "${DEVICE:=cuda:0}"
: "${NPROC_PER_NODE:=4}"          # >1 时使用 torchrun 多卡训练（DDP）
: "${MASTER_PORT:=29600}"         # torchrun 用端口（单机多卡可随意换个空闲端口）
: "${EPOCHS:=10000}"
: "${BATCH_SIZE:=256}"           # per-GPU batch size; global batch = BATCH_SIZE * NPROC_PER_NODE
: "${AUTO_LR:=true}"              # true: 按 global batch 线性缩放 LR（未显式设置 LR 时生效）
: "${BASE_LR:=1e-3}"
: "${BASE_BATCH_SIZE:=1024}"
: "${BASE_NPROC_PER_NODE:=1}"
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

BASE_GLOBAL_BATCH=$((BASE_BATCH_SIZE * BASE_NPROC_PER_NODE))
GLOBAL_BATCH=$((BATCH_SIZE * NPROC_PER_NODE))

if [ "${AUTO_LR,,}" = "true" ] && [ -z "${LR:-}" ]; then
  LR="$(python3 - <<PY
base_lr=float("${BASE_LR}")
gb=int("${GLOBAL_BATCH}")
base_gb=int("${BASE_GLOBAL_BATCH}")
print(f"{base_lr * gb / base_gb:.10g}")
PY
)"
fi
: "${LR:=$BASE_LR}"
echo "Train config: NPROC_PER_NODE=${NPROC_PER_NODE}, BATCH_SIZE=${BATCH_SIZE} (global=${GLOBAL_BATCH}), LR=${LR}, AUTO_LR=${AUTO_LR}"

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

LAUNCH_CMD=(python3 -u index/main.py)
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=""
    for ((i = 0; i < NPROC_PER_NODE; i++)); do
      if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES},${i}"
      else
        CUDA_VISIBLE_DEVICES="${i}"
      fi
    done
    export CUDA_VISIBLE_DEVICES
  fi
  TORCHRUN_ARGS=(--nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")
  if [ -z "${MASTER_ADDR:-}" ]; then
    TORCHRUN_ARGS=(--standalone "${TORCHRUN_ARGS[@]}")
  fi
  LAUNCH_CMD=(torchrun "${TORCHRUN_ARGS[@]}" index/main.py)
fi

${LAUNCH_CMD[@]} \
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
  --wandb_name "$WANDB_RUN_NAME" 
  # > >(tee "$LOG_FILE") 2>&1 &

echo "Index training started. Log file: $LOG_FILE"
echo "W&B Run Name: $WANDB_RUN_NAME"
