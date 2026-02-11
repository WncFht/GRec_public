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
OUTPUT_DIR=./ckpt/Instruments/qwen2.5-3b-instruct-sft__tasks-item2index-seqrec__idx-index_emb-qwen3-embedding-4B_rq4_cb512-512-512-512_dsInstruments_ridJan-28-2026-05-54-58__rid-20260211_1800 \
TASKS=item2index,seqrec \
TRAIN_PROMPT_SAMPLE_NUM=1,1 \
TRAIN_DATA_SAMPLE_NUM=0,0 \
INDEX_FILE=.index_emb-qwen3-embedding-4B_rq4_cb512-512-512-512_dsInstruments_ridJan-28-2026-05-54-58.json \
GPUS=0,1,2,3 \
NPROC=4 \
MASTER_PORT=33326 \
bash scripts/finetune/train_text.sh
```

常用变量：

- 运行资源：`GPUS`, `NPROC`, `MASTER_PORT`
- 数据/模型：`DATASET`, `DATA_PATH`, `BASE_MODEL`, `MODEL_TYPE`, `INDEX_FILE`, `OUTPUT_DIR`
- 默认 `OUTPUT_DIR` 会自动带上 `model/tasks/index/rid`（避免不同实验互相覆盖）
- 脚本会先检查 `data/<dataset>/<dataset><INDEX_FILE>` 是否存在
- 训练超参：`PER_DEVICE_BATCH_SIZE`, `GRAD_ACC`, `LEARNING_RATE`, `EPOCHS`, `NUM_WORKERS`
- 训练行为：`USE_LORA`, `FREEZE`, `USE_GRADIENT_CHECKPOINTING`, `ONLY_TRAIN_RESPONSE`, `DETERMINISTIC`
- 多数据集评估：`EVAL_BY_DATASET=true`, `EVAL_MAIN_DATASET=<dataset>`

> 编译开关说明：`src/finetune/train_ddp.py` 默认开启 `torch.compile()`（Torch>=2.0 且非 Windows）。
> 如需临时回退可设置 `USE_TORCH_COMPILE=false`（`train_text.sh` 会透传该环境变量）。

### 关于 `label_names`（为什么和 `eval_loss` 有关）

- `src/finetune/train_ddp.py` 里现在显式设置了 `TrainingArguments(label_names=["labels"])`。
- 作用：明确告诉 `Trainer` 从 batch 中把 `labels` 作为监督信号传给模型。
- 如果没有这项，在某些 DeepSpeed/DDP 组合下，评估阶段可能只返回 `eval_runtime`/`eval_steps_per_second`，却没有 `eval_loss`。
- 一旦没有 `eval_loss`，`metric_for_best_model=eval_loss` 和 `EarlyStoppingCallback` 就无法正常工作。
- 对应到你的场景：
  - 单数据集：会用 `eval_loss`。
  - `EVAL_BY_DATASET=true` 且 `EVAL_MAIN_DATASET=Instruments`：会用 `eval_Instruments_loss` 做 best metric / early stopping。

---

## 建议

- 新实验优先基于 `train_vl.sh` / `train_text.sh` 启动。
- 需要固化某一组参数时，建议在外层再包一层小脚本，仅设置环境变量后调用统一模板。
- 如需 ZeRO3 合并，继续使用 `convert.sh` 或 `convert/convert.sh`。

---

## 完整参数表

`train_text.sh` / `bundle_train_common.sh` / `bundle_metric_common.sh` 的**全部环境变量**（含默认值与含义）见：

- `docs/finetune_env_vars.md`
