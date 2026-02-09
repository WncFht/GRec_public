# Naming Conventions (Index / SFT / W&B)

本文档说明当前仓库在 `index`、`sft` 和 `wandb` 相关输出的命名规则，便于实验追踪与结果对齐。

## 1. Index 命名规则

### 1.1 Index 训练目录（`index_train_runs`）

Index 训练输出目录结构：

- `index_train_runs/<DATASET>/index/<MODEL_NAME>/<CKPT_TAG>/<RUN_TIME>/...`

其中 `CKPT_TAG` 典型格式：

- `rq{L}_cb{cb-list}_sk{sk-list}_km{kmeans_init}-lkm{large_scale_kmeans}-kmi{kmeans_iters}`

示例：

- `rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100`

### 1.2 Index 生成文件名（`*.index_*.json`）

默认由 `index/generate_indices.py` 自动生成 `output_suffix`，格式为：

- `.index_emb-{embedding}_rq{L}_cb{cb-list}_ds{dataset-tag}_rid{run-id}.json`

最终单数据集文件一般为：

- `<DATASET><output_suffix>`

示例：

- `Instruments.index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-03-50-30.json`

## 2. W&B Run Name 命名规则（SFT）

SFT 的 run name 在 `src/utils.py::make_run_name()` 中自动生成。

当前格式：

- `idx{index_key}__{base_model}__{dataset}__{method}__lr{lr}__b{bs}__gc{0|1}__{tasks}__p{prompt_num}__{MMDD_HHMM}`

字段说明：

- `index_key`：由 `index_file` 去掉 `.index_` 前缀和 `.json` 后缀得到
- `base_model`：`base_model` 路径最后一级目录名
- `dataset`：训练使用的数据集字符串（可多数据集逗号）
- `method`：`Lora` 或 `Finetune`
- `lr`：学习率
- `bs`：`per_device_batch_size`
- `gc`：是否启用 gradient checkpointing（1/0）
- `tasks`：任务列表
- `p`：`train_prompt_sample_num`
- 时间戳：`MMDD_HHMM`

## 3. SFT 输出命名规则

### 3.1 SFT 输出目录

`scripts/finetune/train_text.sh` 默认：

- `./ckpt/{DATASET_TAG}/qwen2.5-3b-sft__idx-{INDEX_KEY}`

其中：

- `DATASET_TAG`：`DATASET` 中逗号替换为 `-`
- `INDEX_KEY`：`INDEX_FILE` 去掉前导 `.`、去掉 `.json`、并将 `/` 替换为 `_`

### 3.2 Checkpoint 子目录

由 HF Trainer 保存，通常为：

- `checkpoint-<global_step>`

保存节奏由以下参数控制：

- `save_and_eval_strategy`
- `save_and_eval_steps`

## 4. 脚本目录命名（finetune bundles）

面向 codebook 实验的 `scripts/finetune` 子目录，统一使用：

- `<DATASET>_<INDEX_PRESET_NAME>`

示例：

- `Instruments_Instruments-qwen3-embedding-4B-rq4_cb32-32-32-32_sk0.0-0.0-0.0-0.003`
- `Instruments_Instruments-Arts-Games-qwen3-embedding-4B-rq3_cb128-128-128_sk0.0-0.0-0.003`

这样可以同时表达：评测数据集 + 对应 index 配置来源。
