export CUDA_VISIBLE_DEVICES=0

# CKPT_PATH=ckpt/Instruments/Llava-onevision-finetune-item2index-index2item-seqrec-fusionseqrec-1-qwen7B/checkpoint-5619
# BASE_MODEL=./ckpt/base_model/llava-onevision-qwen2-7b-ov-hf
# MODEL_TYPE=llava_onevision

# CKPT_PATH=ckpt/Instruments/Qwen2.5-7B-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463
# BASE_MODEL=ckpt/base_model/Qwen2.5-7B
# MODEL_TYPE=qwen

CKPT_PATH=/opt/meituan/dolphinfs_zhangkangning02/zkn/verl/checkpoints/grec_verl/qwen2-vl-7b-n32/global_step_105/actor/merged
BASE_MODEL=ckpt/base_model/Qwen2-VL-7B-Instruct
MODEL_TYPE=qwen2_vl

# DATASET=Arts,Automotive,Cell,Games,Instruments,Pet,Tools,Toys,Sports
# DATASET=Arts,Games,Instruments
DATASET=Instruments

python -m src.seqrec.metric \
    --ckpt_path $CKPT_PATH \
    --model_type $MODEL_TYPE \
    --dataset $DATASET \
    --test_task item2index \
    --test_batch_size 16 \
    --test_prompt_ids 0 \
    --index_file .index_qwen7B.json \
    --ratio_dataset 1 \
    --base_model $BASE_MODEL \
    --num_beams 10
    # --lora \