export CUDA_VISIBLE_DEVICES=0
export CKPT_PATH=./ckpt/qwen_vl_finetune_seqrec/checkpoint-1000
python -m src.infer.test_seqrec_without_itemid --config_file ./config/qwen_vl_finetune_seqrec.yml