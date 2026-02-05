export CUDA_VISIBLE_DEVICES=1
DATASET=Games




python -m data_process.qwen_embeddings \
    --dataset $DATASET