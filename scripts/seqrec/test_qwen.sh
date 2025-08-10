set -e

export CUDA_VISIBLE_DEVICES=0

CONFIG_FILE=./config/qwen_finetune_seqrec_test.yml

echo "Config File: $CONFIG_FILE"

python3 -m src.seqrec.test \
    --config_file $CONFIG_FILE
