export CUDA_VISIBLE_DEVICES=3
export CKPT_PATH=./ckpt/Instruments/Qwen2-VL-2B-Instruct-finetune-seqrec-qwen7B-with-id-0.01/checkpoint-164
export MODEL_TYPE=qwen_2_vl
python -m src.infer.test_with_id --config_file ./config/qwen_vl_finetune_seqrec_without_id_test.yml