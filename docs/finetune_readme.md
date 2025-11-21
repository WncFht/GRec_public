使用 scripts/finetune 下的脚本进行 sft

1. train_ddp_vl_nonewtoken.py 不会添加新 token，训练结果显示最后一个 token 总是学不会
2. train_ddp_vl.py 训练 vlm，会添加新 token
3. train_ddp.py 训练 llm (qwen2.5, llama)
4. train_muon.py 使用 muon 作为优化器，没有继续研究


下面举例说明
```bash
nohup torchrun --nproc_per_node=4 --master_port=33325 -m src.finetune.train_ddp_vl \
    --seed 42 \
    --base_model $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --use_gradient_checkpointing \
    --num_workers 32 \
    --learning_rate 5e-5 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z2_bf16.json \ # 使用 zero2 优化不需要使用 convert.sh 合并 ckpt。如果使用 zero3 的话需要。zero3 会分割训练的模型本身，减少显存占用，但同时也会减慢训练速度
    --bf16 \
    --use_lora \ # 开启 lora 训练
    --lora_modules_to_save "embed_tokens,lm_head" \ # 这里的模块是全量训练的
    --only_train_response \
    --tasks item2index,seqrec,fusionseqrec \
    --train_prompt_sample_num 1,1,1 \ # 一个 data 生成 1 个 sample
    --train_data_sample_num 0,0,0 \ # 是否进行 sample， 0 表示全部取
    --ratio_dataset 1 \ # 是否自取 ratio 的 dataset
    --report_to wandb \
    --index_file .index_qwen7B.json # 使用什么 index
```

batchsize = 4 * 12 * 2