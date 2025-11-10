#!/bin/bash

# Default values
MODE="case"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

MODEL_PATH=./ckpt/Instruments/Qwen2-VL-7B-Instruct-seqrec-item2index-1-qwen7B/checkpoint-84892
MODEL_TYPE=qwen2_vl
BATCH_SIZE=4
IDX_FILE=.index_qwen7B.json
RATIO_DATASET=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --case)
            MODE="case"
            shift
            ;;
        --metric)
            MODE="metric"
            shift
            ;;
        --gpu)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--case | --metric] [--gpu <device_id>]"
            exit 1
            ;;
    esac
done

# Check if mode is specified
if [ -z "$MODE" ]; then
    echo "Error: Please specify either --case or --metric"
    echo "Usage: $0 [--case | --metric] [--gpu <device_id>]"
    exit 1
fi



# Execute based on mode
if [ "$MODE" = "metric" ]; then
    echo "Running metric test..."
    
    python -m src.seqrec.case \
        --ckpt_path $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --test_batch_size $BATCH_SIZE \
        --test_prompt_ids 0 \
        --index_file $IDX_FILE \
        --ratio_dataset $RATIO_DATASET \
        --num_beams 10
        
elif [ "$MODE" = "case" ]; then
    echo "Running case test..."
    
    python -m src.seqrec.metric \
        --ckpt_path $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --test_batch_size $BATCH_SIZE \
        --index_file $IDX_FILE \
        --ratio_dataset $RATIO_DATASET \
        --num_beams 10
fi