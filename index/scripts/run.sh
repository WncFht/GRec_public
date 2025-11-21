DATASET=Instruments
MODEL_NAME=llama
export WANDB_MODE=offline
# DATA_PATH=./data/$DATASET/$DATASET.emb-$MODEL_NAME.npy
# DATA_PATH=./data/Instruments/Qwen/Qwen2.5-VL-7B-Instruct_rep.npy
DATA_PATH=./data/Instruments/Instruments.emb-llama-td.npy

LOG_FILE="./log/index/index_$(date +%Y%m%d%H%M%S).log"

mkdir -p ./log/index
KMEANS_MODE='large'

# --- Logic to set arguments and wandb name based on mode ---

KMEANS_INIT_ARG="true"
LARGE_SCALE_KMEANS_ARG="true"
WANDB_SUFFIX=""

WANDB_RUN_NAME="${DATASET}-${MODEL_NAME}${WANDB_SUFFIX}"
# ----------------------------------------------------

mkdir -p ./log

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
  --data_path $DATA_PATH \
  --ckpt_dir ./data/$DATASET/index/$MODEL_NAME/ \
  --use_wandb False \
  --wandb_project unifymmgrec \
  --wandb_name "$WANDB_RUN_NAME" > "$LOG_FILE" 2>&1 &
  
echo "Indexing started with K-Means Mode: $KMEANS_MODE. Log file: $LOG_FILE"
echo "W&B Run Name: $WANDB_RUN_NAME"