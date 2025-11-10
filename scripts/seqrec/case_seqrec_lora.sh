export CUDA_VISIBLE_DEVICES=1

DATASET=Instruments
RATIO=1

CKPT_PATH=/opt/meituan/dolphinfs_zhangkangning02/zkn/verl/checkpoints/grec_verl/qwen2-vl-7b-n32/global_step_105/actor/merged
BASE_MODEL=./ckpt/base_model/Qwen2-VL-7B-Instruct
MODEL_TYPE=qwen2_vl

# CKPT_PATH=./ckpt/Instruments/Llava-onevision-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463
# BASE_MODEL=./ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
# MODEL_TYPE=llava_onevision

DATA_PATH=./data

python -m src.seqrec.case \
    --model_type $MODEL_TYPE \
    --ckpt_path $CKPT_PATH \
    --base_model $BASE_MODEL \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --test_batch_size 1 \
    --num_beams 10 \
    --ratio_dataset $RATIO \
    --index_file .index_qwen7B.json
    # --lora \