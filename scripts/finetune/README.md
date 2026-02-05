# `scripts/finetune` 使用说明

本目录目前包含两类脚本：

1. **统一模板入口（推荐）**
   - `train_vl.sh`：多模态 SFT 统一入口（`src.finetune.train_ddp_vl`）
   - `train_text.sh`：纯文本 SFT 统一入口（`src.finetune.train_ddp`）
2. **历史实验脚本（保留）**
   - `train_qwen2.5-*.sh`、`train_llava-ov.sh` 等
   - 含个人路径、特定实验配置，仅作为回溯参考

---

## 快速用法

默认后台启动（会输出日志路径与 PID）：

```bash
bash scripts/finetune/train_vl.sh
bash scripts/finetune/train_text.sh
```

调试模式（前台单卡）：

```bash
bash scripts/finetune/train_vl.sh --debug
bash scripts/finetune/train_text.sh --debug
```

---

## 配置方式

统一通过环境变量覆盖默认值（无需改脚本）：

```bash
DATASET=Instruments \
DATA_PATH=./data \
BASE_MODEL=ckpt/base_model/Qwen2.5-3B-Instruct \
MODEL_TYPE=qwen2_5_instruct \
OUTPUT_DIR=./ckpt/Instruments/qwen2.5-3b-sft \
TASKS=item2index,seqrec \
TRAIN_PROMPT_SAMPLE_NUM=1,1 \
TRAIN_DATA_SAMPLE_NUM=0,0 \
INDEX_FILE=.index_qwen3-embedding-4B.json \
GPUS=0,1,2,3 \
NPROC=4 \
MASTER_PORT=33326 \
bash scripts/finetune/train_text.sh
```

常用变量：

- 运行资源：`GPUS`, `NPROC`, `MASTER_PORT`
- 数据/模型：`DATASET`, `DATA_PATH`, `BASE_MODEL`, `MODEL_TYPE`, `INDEX_FILE`, `OUTPUT_DIR`
- 训练超参：`PER_DEVICE_BATCH_SIZE`, `GRAD_ACC`, `LEARNING_RATE`, `EPOCHS`, `NUM_WORKERS`
- 训练行为：`USE_LORA`, `FREEZE`, `USE_GRADIENT_CHECKPOINTING`, `ONLY_TRAIN_RESPONSE`, `DETERMINISTIC`
- 多数据集评估：`EVAL_BY_DATASET=true`, `EVAL_MAIN_DATASET=<dataset>`

---

## 建议

- 新实验优先基于 `train_vl.sh` / `train_text.sh` 启动。
- 需要固化某一组参数时，建议在外层再包一层小脚本，仅设置环境变量后调用统一模板。
- 如需 ZeRO3 合并，继续使用 `convert.sh` 或 `convert/convert.sh`。

