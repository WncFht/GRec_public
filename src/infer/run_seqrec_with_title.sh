export CUDA_VISIBLE_DEVICES=2
export CKPT_PATH=./ckpt/Instruments/Qwen2.5-VL-3B-Instruct-finetune-seqrec-qwen7B-with-title-0.01/checkpoint-41
python -m src.infer.test_seqrec_with_title --config_file ./config/qwen_vl_finetune_seqrec_without_id_test.yml