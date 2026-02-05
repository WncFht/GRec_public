# `scripts/tools` 工具说明

## `gen_run.sh`

用途：按统一命名规则，生成可直接提交到集群的独立运行脚本。

生成内容：

- `runs/generated/<stage>/<yyyymmdd>/<RUN_ID>.sh`
- `runs/specs/<stage>/<yyyymmdd>/<RUN_ID>.env`
- `runs/meta/<stage>/<yyyymmdd>/<RUN_ID>.txt`

支持阶段：

- `sft_text`
- `sft_vl`
- `rl`

---

## 快速示例

### 1) 生成文本 SFT 运行脚本

```bash
bash scripts/tools/gen_run.sh \
  --stage sft_text \
  --dataset Instruments \
  --tasks item2index,seqrec \
  --base-model ckpt/base_model/Qwen2.5-3B-Instruct \
  --model-type qwen2_5_instruct \
  --index-file .index_qwen3-embedding-4B.json \
  --set PER_DEVICE_BATCH_SIZE=8 \
  --set GRAD_ACC=4
```

### 2) 生成 RL 运行脚本

```bash
bash scripts/tools/gen_run.sh \
  --stage rl \
  --dataset Instruments \
  --tasks seqrec \
  --base-model ckpt/Instruments/qwen2.5-sft/checkpoint-1234 \
  --model-type qwen2_5_instruct \
  --index-file .index_qwen3-embedding-4B.json \
  --set REWARD_TYPE=ranking \
  --set NUM_GENERATIONS=16
```

---

## 参数覆盖方式（推荐）

- 通用参数优先使用显式参数（如 `--dataset`、`--tasks`、`--base-model`）。
- 其余训练超参用 `--set KEY=VALUE` 覆盖默认值。
- 所有最终值会写入 `*.env` 快照，便于复现。

常见覆盖项：

- SFT：`PER_DEVICE_BATCH_SIZE`, `GRAD_ACC`, `LEARNING_RATE`, `EPOCHS`, `USE_LORA`
- RL：`TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `NUM_TRAIN_EPOCHS`, `REWARD_TYPE`, `BETA`

---

## 提交建议

生成后直接把 `runs/generated/.../<RUN_ID>.sh` 当作提交对象：

```bash
sbatch --job-name "<RUN_ID>" runs/generated/<stage>/<yyyymmdd>/<RUN_ID>.sh
```

脚本内部会自动设置：

- `WANDB_NAME=<RUN_ID>`
- `WANDB_GROUP=<RUN_KEY>`
- `OUTPUT_DIR=ckpt/<dataset_slug>/<stage>/<RUN_ID>`
- `LOG_FILE=log/<stage>/<dataset_slug>/<yyyymmdd>/<RUN_ID>.log`

