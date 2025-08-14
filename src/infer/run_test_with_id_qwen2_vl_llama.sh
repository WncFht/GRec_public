export CUDA_VISIBLE_DEVICES=5
export CKPT_PATH=./ckpt/Instruments/Qwen2-VL-2B-Instruct-finetune-seqrec-llama-td-with-id-1/checkpoint-10612
export MODEL_TYPE=qwen_2_vl
python -m src.infer.test_with_id --config_file ./config/seqrec_llama-td_test_1.yml