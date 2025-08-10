DATASET=Instruments
DATA_PATH=./data
OUTPUT_DIR=./ckpt/Instruments/Qwen2.5-7B-finetune-seqrec-qwen7B


nohup ./convert/convert.sh $OUTPUT_DIR > convert.log 2>&1 &
