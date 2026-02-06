# Index Pipeline README

本文档描述 `index/`（RQVAE SID 方案）的完整构建与实验追踪流程，重点解决：

- 每次 run 如何可追溯（时间戳目录 + 完整配置）
- 如何系统地做 codebook / sk_epsilons / dataset / embedding model 对比实验
- 如何避免输出文件和实验记录“看起来重复、实际不可复现”

---

## 1. Pipeline 总览

`index/` 方案的标准流程是：

1. 文本转 embedding（可选）：`index/build_embeddings.py`
2. 训练 RQVAE：`index/train_index.py`（通常通过 `index/scripts/train.sh`）
3. 导出索引 json：`index/generate_indices.py`
4. 评估碰撞率与码本利用率：`index/evaluate_index.py`

下游 SFT/RL/Test 统一读取 `Dataset.index_*.json`。

---

## 2. 关键输入与输出

### 2.1 输入

- embedding 文件：`data/<DATASET>/<DATASET>.emb-<MODEL>-td.npy`
-（推荐）row 对应 item_id：`data/<DATASET>/<DATASET>.emb-<MODEL>-td.ids.json`

### 2.2 训练输出目录结构

训练命令里的 `--ckpt_dir` 是“根目录”，真正输出在其时间戳子目录中：

```text
<ckpt_dir>/<timestamp>/
  run_meta.json
  best_loss_model.pth
  best_collision_model.pth
  best_utilization_model.pth
  epoch_*.pth
```

其中 `run_meta.json` 自动保存本次实验关键元信息（参数、数据路径、git commit、机器信息等）。

---

## 3. 推荐用法：`index/scripts/train.sh`

### 3.1 单数据集 vs 多数据集

- `USE_MULTI_DATASETS=true`：使用 `DATASETS` / `DATA_PATHS`（多数据集联合训练）
- `USE_MULTI_DATASETS=false`：使用 `DATA_PATH`（单数据集训练）

### 3.2 Codebook 与 SK 参数

`train.sh` 会自动构建：

- `NUM_EMB_LIST`：每层 codebook 大小
- `SK_EPSILONS`：默认前几层 `0.0`，最后一层 `OPENONEREC_LAST_SK_EPSILON`

可通过环境变量覆盖：

```bash
OPENONEREC_N_LAYERS=4 \
OPENONEREC_CODEBOOK_SIZE=1024 \
OPENONEREC_LAST_SK_EPSILON=0.003 \
bash index/scripts/train.sh
```

### 3.3 自动命名（防冲突）

`train.sh` 默认把以下信息编码进 `CKPT_TAG` / `RUN_NAME`：

- 量化层数与 codebook：`rq*_cb*`
- 各层 `sk_epsilons`：`sk*`
- KMeans 配置：`km*-lkm*-kmi*`

这能显著降低“目录名字不同但配置不清楚”的问题。

---

## 4. Run 可追溯：`run_meta.json`

每个训练 run 会在时间戳目录下自动写入 `run_meta.json`，包含：

- `run_name` / `wandb_name`
- 完整 `args`
- `train_data_paths`
- `git_commit` / `git_branch`
- `host` / `platform` / `python`
- `ckpt_dir`

建议你后续分析实验时，以 `run_meta.json` 作为唯一事实来源，而不是只看目录名。

---

## 5. 导出 index 与评估

### 5.1 导出 index

```bash
python3 -m index.generate_indices \
  --dataset Instruments \
  --ckpt_path <ckpt_path> \
  --output_dir ./data/Instruments \
  --output_file Instruments.index_qwen3-embedding-4B.json \
  --device cuda:0 \
  --batch_size 64
```

### 5.2 评估

```bash
python3 -m index.evaluate_index \
  --ckpt_path <ckpt_path> \
  --device cuda:0 \
  --batch_size 2048
```

---

## 6. 常见实验矩阵建议

你当前最常做的四维度：

1. codebook 大小：`OPENONEREC_CODEBOOK_SIZE`
2. `sk_epsilons`：重点是最后一层 epsilon
3. dataset 组合：`DATASETS=(...)` + `USE_MULTI_DATASETS`
4. embedding model：`MODEL_NAME`

建议固定其余变量，每次只改一个维度，并保证每次 run 都有：

- 唯一的 `RUN_NAME`
- 对应的 `run_meta.json`
- 对应的 index 输出后缀（`OUTPUT_SUFFIX`）

---

## 7. 当前已知限制

- `index/generate_indices.py` 目前默认用行号 `0..N-1` 作为 key；如果你要严格保留原始 item_id，需要额外改造（或使用 `tokenizer/` 路线）。
- `Trainer` 当前仍是单类多职责实现；后续可按 `train/eval/checkpoint` 拆分。

---

## 8. 快速检查清单

训练前：

- 确认 `USE_MULTI_DATASETS` 与 `DATA_PATH` / `DATASETS` 对齐
- 确认 `MODEL_NAME` 与 embedding 文件名一致
- 确认 `OPENONEREC_N_LAYERS` 与 `num_emb_list/sk_epsilons` 预期一致

训练后：

- 检查 `<run_dir>/run_meta.json` 是否存在
- 检查 `best_collision_model.pth` 是否存在
- 导出 index 后，检查输出文件名是否含本次实验标识

