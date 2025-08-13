export CUDA_VISIBLE_DEVICES=0
export CKPT_PATH=/home/fanghaotian/src/GRec/ckpt/base_model/Qwen2.5-VL-3B-Instruct

python -m src.infer.test_without_id --config_file ./config/qwen_vl_finetune_seqrec_withoutid_test.yml