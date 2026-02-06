# Index 数据组织与无冲突使用（SFT / RL）

本文针对 `index/` 路线，目标是：

1. 一个 `INDEX_FILE` 就能看出核心训练配置；
2. SFT / RL 切换不同 index 时不冲突；
3. 目录结构尽量兼容现有代码（不改数据加载逻辑）。

---

## 1. 当前加载约定（必须遵守）

下游代码会按下面规则读文件：

`{data_path}/{dataset}/{dataset}{index_file}`

例如：

- `data_path=./data`
- `dataset=Instruments`
- `index_file=.index_emb-qwen3-embedding-4B_rq4_cb512-512-512-512_dsInstruments_ridJan-28-2026-05-54-58.json`

则必须存在：

`./data/Instruments/Instruments.index_emb-qwen3-embedding-4B_rq4_cb512-512-512-512_dsInstruments_ridJan-28-2026-05-54-58.json`

---

## 2. 推荐 index 命名规范

统一后缀模板：

`.index_emb-<emb>_rq<layers>_cb<cb-list>_ds<train-datasets>_rid<train-id>.json`

字段含义：

- `emb`：训练 index 时使用的 embedding 模型
- `rq`：量化层数（`num_emb_list` 的长度）
- `cb`：每层 codebook 大小（如 `512-512-512-512`）
- `ds`：训练该 index 时使用的数据集组合
- `rid`：训练 run id（通常来自 ckpt 时间戳目录）

> `index/scripts/generate.sh` 默认会自动生成这个后缀；你也可以手动用 `OUTPUT_SUFFIX` 覆盖。

---

## 3. SFT / RL 如何无冲突切换 index

核心原则：

- 每次实验显式设置 `INDEX_FILE`
- 不同 `INDEX_FILE` 使用不同 `OUTPUT_DIR`

示例（SFT）：

```bash
INDEX_FILE=.index_emb-qwen3-embedding-4B_rq4_cb512-512-512-512_dsInstruments_ridJan-28-2026-05-54-58.json \
OUTPUT_DIR=./ckpt/Instruments/sft__idx-rq4-cb512-ridJan-28-2026-05-54-58 \
bash scripts/finetune/train_text.sh
```

示例（RL）：

```bash
INDEX_FILE=.index_emb-qwen3-embedding-4B_rq4_cb1024-1024-1024-1024_dsInstruments_ridFeb-01-2026-10-12-11.json \
OUTPUT_DIR=./ckpt/Instruments/rl__idx-rq4-cb1024-ridFeb-01-2026-10-12-11 \
bash scripts/rl/rl.sh
```

当前模板脚本已默认把 `INDEX_FILE` key 编进 `OUTPUT_DIR` / 日志名，降低混淆概率。

---

## 4. 推荐目录组织（兼容现有代码）

保持现有主结构不变：

```text
data/
  Instruments/
    Instruments.item.json
    Instruments.inter.json
    Instruments.emb-qwen3-embedding-4B-td.npy
    Instruments.emb-qwen3-embedding-4B-td.ids.json
    Instruments.index_emb-..._rid....json
    Instruments.index_emb-..._rid....json
```

建议：

- 一个数据集下可以并存多个 `Instruments.index_*.json`
- 用文件名区分实验，不要复用同名 index 文件
- `best_collision_model.pth` 与 index 文件通过 `rid` 对齐

---

## 5. 更进一步（可选）

如果后续 index 数量非常多，可新增“归档视图”（不改下游读取路径）：

- 例如维护 `data/_index_registry/`，按 `rid` 建软链或清单
- 但 `data/<dataset>/<dataset><index_file>` 仍保留为下游唯一读取入口

这样既兼容旧代码，也方便全局检索。
