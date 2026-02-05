export CUDA_VISIBLE_DEVICES=3
# python src/merge_lora.py \
#   --base_model ckpt/base_model/Qwen2.5-7B \
#   --lora_path ckpt/Instruments/Qwen2.5-7B-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463 \
#   --save_path ckpt/Instruments/Qwen2.5-7B-lora-item2index,seqrec,fusionseqrec-1-qwen7B-merged \
#   --model_type qwen

BASE_MODEL=ckpt/base_model/Qwen2-VL-7B-Instruct
LORA_PATH=ckpt/Instruments/Qwen2-VL-7B-lora-item2index,seqrec,fusionseqrec-1-qwen7B/checkpoint-5463
SAVE_PATH=ckpt/Instruments/Qwen2-VL-7B-lora-item2index-seqrec-fusionseqrec-1-qwen7B-merged
MODEL_TYPE=qwen2_vl
python src/merge_lora.py \
  --base_model $BASE_MODEL \
  --lora_path $LORA_PATH \
  --save_path $SAVE_PATH \
  --model_type $MODEL_TYPE