#!/bin/bash


MODEL_NAME=qwen3-embedding-4B

ROOT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
# 训练好的 ckpt（可以是单数据集训练，也可以是多数据集合并训练的 ckpt）
CKPT_PATH=$ROOT_DIR/GRec/data/Arts-Automotive-Cell-Games-Pet-Sports-Tools-Toys-Instruments/index/qwen3-embedding-4B/rq4_cb1024-1024-1024-1024/Jan-28-2026_05-54-58/epoch_49_collision_0.0576_util_1.0000_model.pth

# 数据根目录（包含各数据集子目录）
DATA_ROOT=$ROOT_DIR/data

# 单数据集（兼容旧用法）：
DATASET=Instruments
DATA_PATH=$DATA_ROOT/$DATASET/${DATASET}.emb-${MODEL_NAME}-td.npy

# 多数据集（推荐）：一次性联合导出，避免不同数据集之间 token 序列碰撞
# 例如：(Arts Games Instruments)
DATASETS=(Arts Automotive Cell Games Pet Sports Tools Toys Instruments)
DATA_PATHS=() # 可选：手动指定多个 .npy 路径；留空则按默认规则拼接
OUTPUT_SUFFIX=".index_${MODEL_NAME}.json"

DEVICE=cuda:0
BATCH_SIZE=64

gen_one() {
  local dataset="$1"
  local data_path="$2"
  local output_dir="$DATA_ROOT/$dataset/"
  local output_file="${dataset}${OUTPUT_SUFFIX}"

  python3 index/generate_indices.py \
    --dataset "$dataset" \
    --ckpt_path "$CKPT_PATH" \
    --data_path "$data_path" \
    --output_dir "$output_dir" \
    --output_file "$output_file" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
}

if [ ${#DATASETS[@]} -gt 0 ]; then
  if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    for d in "${DATASETS[@]}"; do
      DATA_PATHS+=("$DATA_ROOT/$d/${d}.emb-${MODEL_NAME}-td.npy")
    done
  fi

  if [ ${#DATA_PATHS[@]} -ne ${#DATASETS[@]} ]; then
    echo "Error: DATASETS and DATA_PATHS must have the same length."
    exit 1
  fi

  python3 index/generate_indices.py \
    --datasets "${DATASETS[@]}" \
    --ckpt_path "$CKPT_PATH" \
    --data_paths "${DATA_PATHS[@]}" \
    --output_dir "$DATA_ROOT" \
    --output_suffix "$OUTPUT_SUFFIX" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
else
  gen_one "$DATASET" "$DATA_PATH"
fi
