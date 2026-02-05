DATASET=Instruments
DATA_PATH=./data
OUTPUT_DIR=/opt/meituan/dolphinfs_zhangkangning02/zkn/GRec/ckpt/Instruments/Qwen2-VL-2B-Instruct-seqrec,mmitem2index,fusionseqrec-1-qwen7B-5e-5

nohup ./convert/convert.sh $OUTPUT_DIR > convert.log 2>&1 &
