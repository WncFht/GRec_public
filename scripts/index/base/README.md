# Index Scripts README

这份文档是 `index/scripts/` 的速查说明，目标是让你快速回答三个问题：

1. 我现在跑到 pipeline 的哪一步？
2. 这个脚本核心参数是什么？
3. 这次 run 的配置和输出会落在哪里？

---

## 1. 一图看懂 Index Pipeline

推荐顺序（`index/` 这条 SID 路线）：

1. `text2emb.sh`：从 `*.item.json` 抽取 embedding（并生成 `*.ids.json`）
2. `train.sh`：训练 RQVAE index 模型（输出 checkpoint + `run_meta.json`）
3. `generate.sh`：把 item 映射为离散 token 序列，导出 `Dataset.index_*.json`
4. `evaluate.sh`：评估 checkpoint（碰撞率 / 利用率等）

另外两个脚本是训练包装器：

- `train_nohup.sh`：后台训练包装
- `train_kmeans_ablation.sh`：KMeans 消融包装（`large/small/none`）

---

## 2. 脚本清单（职责）

- `train.sh`：主训练入口（推荐直接使用）
- `train_nohup.sh`：给 `train.sh` 注入常用 env 并后台启动
- `train_kmeans_ablation.sh`：切换 KMeans 配置并后台启动 `train.sh`
- `text2emb.sh`：多数据集 embedding 抽取
- `generate.sh`：导出 index json（支持单/多数据集）
- `evaluate.sh`：评估某个 checkpoint

---

## 3. 训练（核心）`train.sh`

### 3.1 关键环境变量

数据相关：

- `ROOT_DIR`：数据根上级目录（默认服务器路径）
- `MODEL_NAME`：embedding 模型名（用于拼接默认数据路径）
- `USE_MULTI_DATASETS`：`true/false`
- `DATASET`、`DATA_PATH`：单数据集模式
- `DATASETS`、`DATA_PATHS`：多数据集模式（空格分隔）

量化相关：

- `INDEX_N_LAYERS`：RQVAE 层数
- `INDEX_CODEBOOK_SIZE`：每层 codebook 大小
- `INDEX_LAST_SK_EPSILON`：最后一层 `sk_epsilon`
- `INDEX_KMEANS_ITERS`：KMeans 迭代次数
- `KMEANS_INIT_ARG`、`LARGE_SCALE_KMEANS_ARG`：KMeans 初始化策略

训练相关：

- `NPROC_PER_NODE`、`MASTER_PORT`
- `BATCH_SIZE`、`EPOCHS`
- `AUTO_LR`、`BASE_LR`
- `LR`（若手动指定会覆盖自动缩放）
- `USE_WANDB`、`WANDB_PROJECT`、`WANDB_RUN_NAME`

输出相关：

- `INDEX_TRAIN_ROOT`：index 训练产物根目录（默认 `./index_train_runs`）
- `CKPT_TAG`：run 命名标签（默认自动拼接）
- `CKPT_DIR`：checkpoint 输出目录（默认 `${INDEX_TRAIN_ROOT}/<dataset>/index/<model>/<ckpt_tag>/`）
- `LOG_FILE`：训练日志文件路径

### 3.2 常用示例

单数据集：

```bash
USE_MULTI_DATASETS=false \
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
NPROC_PER_NODE=1 \
BATCH_SIZE=2048 \
bash scripts/index/base/train.sh
```

多数据集：

```bash
USE_MULTI_DATASETS=true \
DATASETS="Arts Automotive Cell Games Pet Sports Tools Toys Instruments" \
MODEL_NAME=qwen3-embedding-4B \
INDEX_CODEBOOK_SIZE=1024 \
INDEX_N_LAYERS=4 \
INDEX_LAST_SK_EPSILON=0.003 \
NPROC_PER_NODE=4 \
BATCH_SIZE=256 \
bash scripts/index/base/train.sh
```

指定自定义 run 标签（推荐）：

```bash
CKPT_TAG="rq4_cb1024_sk0-0-0-0.003_expA" \
WANDB_RUN_NAME="rq4-expA" \
bash scripts/index/base/train.sh
```

指定 index 模型统一存储根目录（避免混在 `data/`）：

```bash
INDEX_TRAIN_ROOT=./index_train_runs \
bash scripts/index/base/train.sh
```

---

## 4. 后台训练包装器

### 4.1 `train_nohup.sh`

用途：在不改 `train.sh` 逻辑的前提下，快速后台启动训练。

```bash
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
BATCH_SIZE=2048 \
LR=1e-4 \
bash index/scripts/train_nohup.sh
```

### 4.2 `train_kmeans_ablation.sh`

用途：快速做 KMeans 消融。

- `KMEANS_MODE=large`：`kmeans_init=true`, `large_scale_kmeans=true`
- `KMEANS_MODE=small`：`kmeans_init=true`, `large_scale_kmeans=false`
- `KMEANS_MODE=none`：`kmeans_init=false`, `large_scale_kmeans=false`

```bash
KMEANS_MODE=none \
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
bash index/scripts/train_kmeans_ablation.sh
```

---

## 5. Embedding 抽取 `text2emb.sh`

### 5.1 输入与输出

输入：`data/<DATASET>/<DATASET>.item.json`

输出：

- `data/<DATASET>/<DATASET>.emb-<PLM_NAME>-td.npy`
- `data/<DATASET>/<DATASET>.emb-<PLM_NAME>-td.ids.json`

### 5.2 常用示例

```bash
DATASETS="Instruments Toys" \
PLM_NAME="Llama-3.1-8B-Instruct" \
MODEL_PATH="/path/to/your/plm" \
NUM_PROCESSES=4 \
BATCH_SIZE=256 \
FORCE_REBUILD=0 \
bash index/scripts/text2emb.sh
```

备注：

- `FORCE_REBUILD=0` 且 embedding 已存在时，会跳过；若只缺 ids，会自动补 ids。
- `TMP_DIR` 可用于指定中间文件目录（大数据集时建议放高速盘）。

---

## 6. 导出 index `generate.sh`

### 6.1 关键点

- 必填：`CKPT_PATH`
- 支持单数据集和多数据集
- 默认输出后缀：自动命名（包含 emb/rq/cb/ds/rid）
- 命名模板：`.index_emb-<emb>_rq<layers>_cb<cb-list>_ds<train-datasets>_rid<train-id>.json`
- 若你要强制自定义后缀，可显式设置 `OUTPUT_SUFFIX`

单数据集：

```bash
USE_MULTI_DATASETS=false \
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
CKPT_PATH=/path/to/best_collision_model.pth \
bash scripts/index/base/generate.sh
```

如果不传 `OUTPUT_SUFFIX`，脚本会自动生成包含训练关键信息的后缀；下游只需把该后缀作为 `INDEX_FILE` 传给 SFT/RL。

多数据集：

```bash
USE_MULTI_DATASETS=true \
DATASETS="Arts Automotive Cell Games Pet Sports Tools Toys Instruments" \
MODEL_NAME=qwen3-embedding-4B \
CKPT_PATH=/path/to/best_collision_model.pth \
bash scripts/index/base/generate.sh
```

如需手工覆盖命名：

```bash
OUTPUT_SUFFIX=.index_my_exp_tag.json \
CKPT_PATH=/path/to/best_collision_model.pth \
bash scripts/index/base/generate.sh
```

---

## 7. 评估 `evaluate.sh`

推荐直接给 `CKPT_PATH`（最清晰）：

```bash
CKPT_PATH=/path/to/best_collision_model.pth \
DEVICE=cuda:0 \
BATCH_SIZE=2048 \
bash index/scripts/evaluate.sh
```

也支持 `CKPT_BASE_DIR + MODEL_FILE`，或 `TIMESTAMP` 自动拼接（默认按 `INDEX_TRAIN_ROOT/<dataset>/index/<model>/<timestamp>`）。

---

## 8. 实验管理建议（避免 run 混乱）

- 每次实验显式设置 `CKPT_TAG` 或 `WANDB_RUN_NAME`
- 训练后优先读取时间戳目录里的 `run_meta.json` 做真实配置记录
- 一个实验只改一个主维度（codebook / sk_epsilon / dataset 组合 / embedding 模型）
- 多数据集实验固定 `DATASETS` 顺序，避免目录名和结果对不上

---

## 9. 常见坑

- `DATASETS` 和 `DATA_PATHS` 是空格分隔，不是逗号分隔
- `generate.sh` 不传 `CKPT_PATH` 会直接报错退出
- `evaluate.sh` 的默认 `MODEL_NAME=qwen7B` 仅是默认值，建议显式覆盖或直接传 `CKPT_PATH`
- `run_meta.json` 才是配置真相，不要只看目录时间戳
