# Naming Conventions (Index / SFT / Metric / W&B)

本文档定义当前仓库在 `index`、`sft`、`metric`、`wandb` 的命名规范。

目标：

- 同一实验的 `output_dir / checkpoint / wandb / metric result` 一眼可对齐
- 多组并行训练（如 4 卡/8 卡、不同 lr/epoch/batch）不互相覆盖
- 命名可扩展且可读

> 说明：本规范刻意**不包含 git sha**，以便命名更短、使用更直观。

---

## 1. 通用命名原则

### 1.1 规范化（tag/slug）

建议统一使用以下规则处理字段：

- 空格删除或替换为 `-`
- 逗号替换为 `-`
- 路径分隔符 `/` 替换为 `_`
- 保留大小写（当前脚本多数保持原样），但同一字段内应一致

### 1.2 关键字段

- `DATASET_TAG`：`DATASET` 的逗号版本（如 `Arts,Games` → `Arts-Games`）
- `TASKS_TAG`：`TASKS` 的逗号版本（如 `item2index,seqrec` → `item2index-seqrec`）
- `INDEX_KEY`：`INDEX_FILE` 去前导 `.`、去 `.json`、并将 `/` 替换为 `_`
- `RUN_ID`：本次运行唯一标识（默认时间戳 `YYYYmmdd_HHMMSS`）

---

## 2. Index 命名规则

### 2.1 Index 训练目录（`index_train_runs`）

Index 输出目录结构：

- `index_train_runs/<DATASET>/index/<MODEL_NAME>/<CKPT_TAG>/<RUN_TIME>/...`

`CKPT_TAG` 典型格式：

- `rq{L}_cb{cb-list}_sk{sk-list}_km{kmeans_init}-lkm{large_scale_kmeans}-kmi{kmeans_iters}`

示例：

- `rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100`

### 2.2 Index 文件名（`*.index_*.json`）

常见后缀格式：

- `.index_emb-{embedding}_rq{L}_cb{cb-list}_ds{dataset-tag}_rid{run-id}.json`

最终文件名：

- `<DATASET><output_suffix>`

示例：

- `Instruments.index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11.json`

---

## 3. SFT 命名规则（当前已落地）

### 3.1 `train_text.sh` 默认输出目录

当前默认格式：

- `ckpt/{DATASET_TAG}/qwen2.5-3b-sft__tasks-{TASKS_TAG}__idx-{INDEX_KEY}__rid-{RUN_ID}`

示例：

- `ckpt/Instruments/qwen2.5-3b-sft__tasks-item2index-seqrec-fusionseqrec__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11__rid-20260211_021530`

该格式用于避免重复训练覆盖（`RUN_ID` 每次不同）。

### 3.2 Checkpoint 子目录

HF Trainer 默认保存：

- `checkpoint-<global_step>`

保存节奏由以下参数决定：

- `save_and_eval_strategy`
- `save_and_eval_steps`

---

## 4. Metric 命名规则（当前已落地）

对于 `Instruments_*` 的 bundle metric 脚本：

- 若未显式传入 `CKPT_PATH`，会按 `TASKS + INDEX_KEY` 自动定位最新训练目录：
  - `ckpt/$DATASET/qwen2.5-3b-sft__tasks-{TASKS_TAG}__idx-{INDEX_KEY}__rid-*`
- 再从该目录选取最新 `checkpoint-*`

结果目录结构：

- `results/<split>/{TASK}-constrained/{DATASET}-{INDEX_TAG}/{MODEL_DIR}/{CHECKPOINT_NAME}`

该规则保证 train 与 metric 可通过 `tasks + index` 对齐。

---

## 5. W&B 命名规则（SFT）

### 5.1 项目归属

- `WANDB_ENTITY=generate_rec`
- `WANDB_PROJECT=grec_sft`

### 5.2 Run 名称

当前默认建议：

- `WANDB_NAME=sft_text_{DATASET_TAG}__tasks-{TASKS_TAG}__idx-{INDEX_KEY}`

说明：

- `WANDB_NAME` 用于可读性和检索
- 具体唯一性主要依赖 `output_dir` 的 `RUN_ID`

如果希望 W&B 中也唯一，可在自定义时追加 `__rid-{RUN_ID}`。

---

## 6. 面向超参扩展的推荐命名（建议）

当后续频繁对比 `epoch / learning_rate / batch_size / nproc(4卡→8卡)` 时，建议在命名中加入训练关键签名。

### 6.1 推荐新增字段

- `lr{LEARNING_RATE}`
- `ep{EPOCHS}`
- `b{PER_DEVICE_BATCH_SIZE}`
- `ga{GRAD_ACC}`
- `n{NPROC}`
- `ebs{effective_batch_size}`，其中：
  - `effective_batch_size = per_device_batch_size * grad_acc * nproc`

### 6.2 推荐放置位置

- `WANDB_NAME`：追加在 `tasks` 与 `idx` 之间或末尾
- `OUTPUT_DIR`：建议追加在 `tasks` 后面，`idx` 前面

示例：

- `...__tasks-item2index-seqrec__b2ga2n8ebs32__idx-...__rid-20260211_103011`

---

## 7. Bundle 目录命名

面向 codebook 实验的 `scripts/finetune` 子目录保持：

- `<DATASET>_<INDEX_PRESET_NAME>`

示例：

- `Instruments_Instruments-qwen3-embedding-4B-rq4_cb32-32-32-32_sk0.0-0.0-0.0-0.003`
- `Instruments_Instruments-qwen3-embedding-4B-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003`

该命名用于表达“评测数据集 + index 配置来源”。

---

## 8. 最小落地检查清单

每次新增实验脚本时建议自检：

- `OUTPUT_DIR` 是否包含：`tasks / idx / rid`
- `metric` 自动找 ckpt 是否使用同一 `tasks + idx` 规则
- `WANDB_ENTITY/PROJECT` 是否为 `generate_rec/grec_sft`
- 是否可以在不改脚本代码的情况下，仅通过环境变量覆盖关键参数
