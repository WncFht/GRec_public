export CUDA_VISIBLE_DEVICES=4
export CKPT_PATH=./ckpt/Instruments/Qwen2-VL-2B-Instruct-finetune-seqrec-qwen7B-with-id-0.1-freeze/checkpoint-1592
export MODEL_TYPE=llava_onevision
python -m src.infer.test_seqrec_with_item_id --config_file ./config/qwen_vl_finetune_seqrec_without_id_test.yml