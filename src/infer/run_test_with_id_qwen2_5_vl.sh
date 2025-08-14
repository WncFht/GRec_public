export CUDA_VISIBLE_DEVICES=5
export CKPT_PATH=./ckpt/Instruments/Qwen2.5-VL-3B-Instruct-finetune-seqrec-qwen7B-with-id-0.1/checkpoint-796
export MODEL_TYPE=qwen2_5_vl
python -m src.infer.test_with_id --config_file ./config/qwen_vl_finetune_seqrec_with_id_test_0.1.yml