#!/bin/bash

# === Configuration ===
# 1. The dataset name, used to locate the model checkpoint.
DATASET=Instruments

# 2. The model name, part of the directory structure.
# MODEL_NAME=llama-td
MODEL_NAME=qwen7B
# 3. The specific trained model file to evaluate.
# You can choose: best_loss_model.pth, best_collision_model.pth, best_utilization_model.pth, or a specific epoch's model.
MODEL_FILE=best_collision_model.pth

# 4. The base directory where model checkpoints are stored.
#    This needs to match the structure from your training run. It usually includes a timestamp.
#    PLEASE UPDATE THIS PATH to point to your actual training output directory.
#    Example: CKPT_BASE_DIR=./data/Instruments/index/llama-td/Jul-10-2024_10-30-00
# TIMESTAMP=Jul-09-2025_15-16-54
TIMESTAMP=Jul-08-2025_13-00-53
CKPT_BASE_DIR=./data/$DATASET/index/$MODEL_NAME/$TIMESTAMP

# 5. The device to run evaluation on.
DEVICE=cuda:0

# 6. Batch size for evaluation. Use a larger value if your GPU memory allows.
BATCH_SIZE=2048
# =====================


# --- DO NOT EDIT BELOW THIS LINE ---
# Construct the full path to the checkpoint
CKPT_PATH=${CKPT_BASE_DIR}/${MODEL_FILE}

# Check if the checkpoint file exists
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found at '$CKPT_PATH'"
    echo "Please ensure the 'DATASET', 'MODEL_NAME', 'TIMESTAMP', and 'MODEL_FILE' variables are set correctly in evaluate.sh."
    exit 1
fi

echo "======================================================"
echo "Starting evaluation for checkpoint: $CKPT_PATH"
echo "Device: $DEVICE, Batch Size: $BATCH_SIZE"
echo "======================================================"

python3 index/evaluate_index.py \
  --ckpt_path "$CKPT_PATH" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE"

echo "======================================================"
echo "Evaluation finished."
echo "======================================================" 