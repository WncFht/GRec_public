#!/bin/bash

MODEL_NAME=llama

# 训练好的 ckpt（可以是单数据集训练，也可以是多数据集合并训练的 ckpt）
CKPT_PATH=./data/Instruments/index/$MODEL_NAME/Jul-24-2025_14-55-51/best_collision_model.pth

# 单数据集（兼容旧用法）：
DATASET=Instruments
DATA_PATH=./data/$DATASET/${DATASET}.emb-${MODEL_NAME}-td.npy

# 多数据集：把 DATASETS 填上，然后会按每个数据集分别生成 index 并落盘到各自目录
DATASETS=() # 例如：(Arts Games Instruments)
DATA_PATHS=() # 可选：手动指定多个 .npy 路径；留空则按默认规则拼接

DEVICE=cuda:0
BATCH_SIZE=64

gen_one() {
  local dataset="$1"
  local data_path="$2"
  local output_dir="./data/$dataset/index/"
  local output_file="${dataset}.index_${MODEL_NAME}.json"

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
      DATA_PATHS+=("./data/$d/${d}.emb-${MODEL_NAME}-td.npy")
    done
  fi

  if [ ${#DATA_PATHS[@]} -ne ${#DATASETS[@]} ]; then
    echo "Error: DATASETS and DATA_PATHS must have the same length."
    exit 1
  fi

  for i in "${!DATASETS[@]}"; do
    gen_one "${DATASETS[$i]}" "${DATA_PATHS[$i]}"
  done
else
  gen_one "$DATASET" "$DATA_PATH"
fi
