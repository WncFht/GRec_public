DATASET=Instruments
MODEL_NAME=llama
CKPT_PATH=./data/$DATASET/index/$MODEL_NAME/Jul-24-2025_14-55-51/best_collision_model.pth
# dolphinfs_zhangkangning02/zkn/unnifymmgrec/data/Instruments/index/llama/Jul-24-2025_14-55-51
OUTPUT_DIR=./data/$DATASET/index/
OUTPUT_FILE=${DATASET}.index_${MODEL_NAME}.json
DEVICE=cuda:0

python3 index/generate_indices.py \
  --dataset $DATASET \
  --ckpt_path $CKPT_PATH \
  --output_dir $OUTPUT_DIR \
  --output_file $OUTPUT_FILE \
  --device $DEVICE \
  --batch_size 64