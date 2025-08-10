#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# --- 配置 ---
DATASET=Instruments
DATA_PATH=./data
MODEL_ALISE=llama-7b-lora
BASE_MODEL_DIR=/hpc_stor03/sjtu_home/haotian.fang/.cache/modelscope/hub/models/skyline2006/llama-7b
CKPT_PATH=./ckpt/$DATASET/$MODEL_ALISE
RESULTS_FILE=./results/$DATASET/${MODEL_ALISE}-seqrec-test-legacy.json

# --- 准备工作 ---
mkdir -p "$(dirname "$RESULTS_FILE")"
echo "--- 启动 LLaMA LoRA 模型测试 (使用 test.py) ---"
echo "  数据集: $DATASET"
echo "  检查点: $CKPT_PATH"
echo "  结果将保存至: $RESULTS_FILE"
echo "-----------------------------------"

# --- 执行测试 ---
python3 src/test.py \
    --gpu_id 0 \
    --base_model "$BASE_MODEL_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --lora \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --results_file "$RESULTS_FILE" \
    --test_batch_size 8 \
    --num_beams 20 \
    --test_task seqrec \
    --test_prompt_ids 0 \
    --index_file .index_llama-td.json \
    --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10"
