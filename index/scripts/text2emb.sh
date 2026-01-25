#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
MODEL_PATH="$HOME_DIR/ckpt/base_model/Qwen3-Embedding-4B"
: "${PLM_NAME:=qwen3-embedding-4B}"
: "${NUM_PROCESSES:=4}"
: "${BATCH_SIZE:=256}"   # will auto-reduce on OOM
: "${MAX_SENT_LEN:=2048}"
: "${FORCE_REBUILD:=0}"  # set to 1 to overwrite existing embeddings
: "${TMP_DIR:=}"         # optional, e.g. /tmp (faster/safer than network FS)

# Helps avoid CUDA memory fragmentation on long runs.
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

# 剩余的 dataset；如果需要把 Instruments 也重跑，就在这里加上 Instruments
datasets=(Arts Automotive Cell Games Pet Sports Tools Toys Instruments)

for dataset in "${datasets[@]}"; do
    DATASET_DIR="$HOME_DIR/data/$dataset"
    echo "========== Start $dataset =========="
    echo "ROOT: $DATASET_DIR"

    OUT_EMB="$DATASET_DIR/${dataset}.emb-${PLM_NAME}-td.npy"
    OUT_IDS="$DATASET_DIR/${dataset}.emb-${PLM_NAME}-td.ids.json"

    if [[ "$FORCE_REBUILD" != "1" && -f "$OUT_EMB" ]]; then
        if [[ -f "$OUT_IDS" ]]; then
            echo "Skip $dataset (embedding exists): $OUT_EMB"
            continue
        fi
        echo "Embedding exists but ids missing, generating ids only: $OUT_IDS"
        python3 index/text2ids.py \
            --dataset "$dataset" \
            --root "$DATASET_DIR" \
            --plm_name "$PLM_NAME" \
            --emb_path "$OUT_EMB"
        echo "========== Done  $dataset (ids only) =========="
        continue
    fi

    RUN_ID="$(date +%s%N)"
    TMP_ARGS=()
    if [[ -n "$TMP_DIR" ]]; then
        TMP_ARGS+=(--tmp_dir "$TMP_DIR")
    fi
    accelerate launch --num_processes "$NUM_PROCESSES" index/text2emb.py \
        --dataset "$dataset" \
        --root "$DATASET_DIR" \
        --plm_name "$PLM_NAME" \
        --run_id "$RUN_ID" \
        --batch_size "$BATCH_SIZE" \
        --max_sent_len "$MAX_SENT_LEN" \
        "${TMP_ARGS[@]}" \
        --plm_checkpoint "$MODEL_PATH"

    echo "========== Done  $dataset =========="
done
