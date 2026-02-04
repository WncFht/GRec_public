# 数据处理 / Embedding 抽取

本项目的数据处理阶段主要包含：文本 enrich、图片下载、（多模态）embedding 抽取等。大多数脚本在 `data_process/` 下。

---

## 1. 输入数据约定（以当前抽取脚本为准）

以 `data_process/qwen_embeddings.py` 的实现为准，embedding 抽取默认读取：

```
data/<DATASET>/
  <DATASET>.item2id                  # item_id ↔ num_id（tab 分隔）
  <DATASET>.item_enriched_v2.json     # item 信息（含原始/增强文本字段）
  images/<item_id>.jpg                # 图片文件（可选；无图会自动降级为 text-only）
```

> 说明：目前 `qwen_embeddings.py` 写死读取 `<DATASET>.item_enriched_v2.json`；如果你只有 `<DATASET>.item.json`，需要先按你的流程生成 enriched 版本或改代码里的文件名。

---

## 2. 抽取单个配置的 embedding（推荐入口）

直接运行：

```bash
python data_process/qwen_embeddings.py \
  --dataset Instruments \
  --model ckpt/base_model/Qwen2-VL-7B-Instruct \
  --out-dir reps \
  --batch-size 8 \
  --mode orig_enhanced
```

常用选项：

- `--mode`：`orig` / `enhanced` / `orig_enhanced`
- `--no-image`：强制 text-only
- `--image-only`：强制 image-only（无图会用占位文本兜底，避免空输入）

输出默认在 `data/<DATASET>/<out-dir>/`：

- `*.json`：每个 `num_id` 一条 `{representation, text, has_image, ...}`
- `*.npy`：按 `num_id` 排序后的矩阵 `[N, D]`
- `*.texts.txt`：抽样保存的构造文本（便于快速核对 mode 差异）

---

## 3. 一次性生成多种组合（便捷脚本）

`scripts/extract_rep.py` 是一个 wrapper，会自动生成多种 “文本 mode × 是否带图 + image_only” 的组合：

```bash
python scripts/extract_rep.py \
  --dataset Instruments \
  --model ckpt/base_model/Qwen2.5-VL-3B-Instruct \
  --out-dir reps \
  --modes orig,orig_enhanced,enhanced
```

默认会生成：`image_only(img)` + 每个 mode 的 `img/noimg`，总计 `1 + 2 * len(modes)` 份输出。

---

## 4. 备注

- `data_process/qwen_embedding_batch.py` 目前与 `qwen_embeddings.py` 的功能不完全对齐；如无特殊需求，优先使用 `qwen_embeddings.py` 或 `scripts/extract_rep.py`。

