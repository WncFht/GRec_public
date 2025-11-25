# Lab

## LC-Rec

### Arts

`https://wandb.ai/wncfht/LC-REC/runs/xlxgjmto`

```bash
export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export WANDB_PROJECT=LC-REC
export WANDB_NAME=Arts-bs4-8-4-index_llama-seqrec-item2index-index2item-fusionseqrec-30000
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1,2,3


DATASET=Arts
BASE_MODEL=ckpt/base_model/llama-7b
DATA_PATH=data
OUTPUT_DIR=ckpt/$DATASET/LC-Rec

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Use 'tail -f $LOG_FILE' to monitor progress"


nohup torchrun --nproc_per_node=4 --master_port=33324 finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z2_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item,fusionseqrec \
    --train_prompt_sample_num 1,1,1,1 \
    --train_data_sample_num 0,0,0,30000 \
    --index_file .index_lemb.json > $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
echo "Training started with PID: $PID"
echo "To stop training: kill $PID"

# 保存PID到文件以便后续管理
echo $PID > $OUTPUT_DIR/training.pid

tail -f $LOG_FILE
```

4 epoch 收敛，loss=1.5259

```text
Prompt 0 results:  {'hit@1': 0.06798087333092746, 'hit@5': 0.10113677372789606, 'hit@10': 0.12545110068567306, 'ndcg@5': 0.08487453582060851, 'ndcg@10': 0.09268070690124625}
```

## Games

`https://wandb.ai/wncfht/LC-REC/runs/eic3lzfw`

```bash
export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export WANDB_PROJECT=LC-REC
export WANDB_NAME=Games-bs4-8-4-index_llama-seqrec-item2index-index2item-fusionseqrec-30000
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1,2,3


DATASET=Games
BASE_MODEL=ckpt/base_model/llama-7b
DATA_PATH=data
OUTPUT_DIR=ckpt/$DATASET/LC-Rec

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Use 'tail -f $LOG_FILE' to monitor progress"


nohup torchrun --nproc_per_node=4 --master_port=33324 finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z2_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item,fusionseqrec \
    --train_prompt_sample_num 1,1,1,1 \
    --train_data_sample_num 0,0,0,100000 \
    --index_file .index_lemb.json > $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
echo "Training started with PID: $PID"
echo "To stop training: kill $PID"

# 保存PID到文件以便后续管理
echo $PID > $OUTPUT_DIR/training.pid

tail -f $LOG_FILE
```

```text
Prompt 0 results:  {'hit@1': 0.022860658841347973, 'hit@5': 0.0649375236652783, 'hit@10': 0.09953616054524801, 'ndcg@5': 0.04394149921659115, 'ndcg@10': 0.055049765659543094}
```
