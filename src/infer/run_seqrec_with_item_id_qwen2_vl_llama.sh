export CUDA_VISIBLE_DEVICES=4
export CKPT_PATH=./ckpt/Instruments/Qwen2-VL-2B-Instruct-finetune-seqrec-llama-td-with-id-1/checkpoint-10612
export MODEL_TYPE=qwen_2_vl
python -m src.infer.test_seqrec_with_item_id --config_file ./config/seqrec_llama-td_test_1.yml