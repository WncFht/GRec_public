# SID（离散索引）构建指南

> 注意：所有命令默认在 `src/GRec/` 项目根目录下运行（即包含 `index/`、`tokenizer/`、`src/` 的目录）。

SID（Semantic/Structured Indexing）在 GRec 里指：把连续的 item embedding（`.npy`）量化成**离散 token 序列**，
并导出 `Dataset.index_*.json`，供下游 **SFT / RL / 测试**阶段通过 `--index_file` 加载。

目前 SID 有两套实现（你提到的两种方式）：

- **方式 A：`index/`（RQVAE / 深度模型）**：训练一个 Encoder→ResidualVQ→Decoder，把 embedding 映射到 codes，适合追求更强表达能力（训练更重）。
- **方式 B：`tokenizer/`（Residual K-Means / OpenOneRec-style）**：逐层 residual KMeans 学 codebook，轻量、稳定，且天然适合“多数据集共享同一个 tokenizer/codebook”。

下游数据加载逻辑对两者完全一致：都输出 `Dataset.index_*.json`（key=item_id，value=token list）。

---

## 0. 输出格式与命名约定（非常关键）

### 0.1 输出 JSON 结构

两种 SID 都导出形如：

```json
{
  "0": ["<a_12>", "<b_7>", "<c_1024>"],
  "1": ["<a_88>", "<b_3>", "<c_2047>"]
}
```

- key：`item_id`（字符串）
- value：token 列表，训练时会用 `''.join(tokens)` 得到最终的 item token 串（例如 `"<a_12><b_7><c_1024>"`）

### 0.2 `--index_file` 的拼接规则

GRec 读取索引文件的路径是：

`{data_path}/{dataset}/{dataset}{index_file}`

因此：

- 你传 `--data_path ./data --dataset Instruments --index_file .index_xxx.json`
- 则文件必须存在：`./data/Instruments/Instruments.index_xxx.json`

常见坑：`--index_file` 要带开头的点 `.`，并且要以 `.json` 结尾（脚本里多数都默认这样命名）。

---

## 1. 输入准备：embedding 与 id 对齐

SID 的输入是一个二维 `.npy`，形状 `[num_items, dim]`。重要的是 **npy 的第 i 行代表哪个 item_id**：

- 如果你的数据集 item_id 本身就是 `0..N-1` 且与 `.inter.json/.item.json` 的 ID 体系一致，那么直接用行号即可；
- 如果你的 item_id 不是连续整数（或来自原始 asin/pid），强烈建议同时保存一份 `*.ids.json`，明确每一行对应的 item_id。

### 1.1 用 `index/text2emb.py` 生成 text embedding（推荐有 ids）

`index/text2emb.py` 会从 `data/<DATASET>/<DATASET>.item.json` 读取文本字段（title/description），抽取 embedding，并输出：

- `data/<DATASET>/<DATASET>.emb-<PLM_NAME>-td.npy`
- `data/<DATASET>/<DATASET>.emb-<PLM_NAME>-td.ids.json`

常用批处理脚本：`index/scripts/text2emb.sh`（accelerate 多进程 + 文件方式 merge）。

---

## 2. 方式 A：`index/`（RQVAE / 深度 SID）

入口与脚本：

- 训练：`index/main.py`（可用 `index/scripts/run.sh`/`train.sh` 包装）
- 生成 index json：`index/generate_indices.py`（可用 `index/scripts/generate.sh`）
- 评估：`index/evaluate_index.py`（可用 `index/scripts/evaluate.sh`）
- 详细架构说明：`index/README.md`

### 2.1 数据要求

- 输入 `.npy` 必须是二维数组 `[num_items, dim]`
- 支持单数据集 `--data_path`，也支持多数据集合并训练 `--data_paths a.npy b.npy ...`（内部 `MultiEmbDataset` 会把它们视作一个大 dataset）

### 2.2 训练 RQVAE

最常用的几个参数：

- `--data_path` / `--data_paths`：embedding 文件
- `--ckpt_dir`：输出目录（脚本会在其下创建时间戳子目录）
- `--num_emb_list`：每层 codebook 大小（列表长度=量化层数）
- `--layers`、`--e_dim`：Encoder/Decoder 结构与 latent 维度
- `--kmeans_init`、`--large_scale_kmeans`：是否用 KMeans 初始化 codebook（`index/` 使用 sklearn KMeans，CPU）

示例（请按实际路径改）：

```bash
python3 index/main.py \
  --data_path ./data/Instruments/Instruments.emb-qwen3-embedding-4B-td.npy \
  --ckpt_dir  ./data/Instruments/index/rqvae_qwen3-embedding-4B \
  --num_emb_list 8192 8192 8192 \
  --layers 2048 1024 512 256 128 64 \
  --e_dim 32 \
  --device cuda:0
```

### 2.3 生成 `Dataset.index_*.json`

```bash
python3 index/generate_indices.py \
  --dataset Instruments \
  --ckpt_path  ./data/Instruments/index/rqvae_qwen3-embedding-4B/<TIMESTAMP>/best_collision_model.pth \
  --output_dir ./data/Instruments \
  --output_file Instruments.index_rqvae_qwen3-embedding-4B.json \
  --device cuda:0 \
  --batch_size 64
```

脚本会检测碰撞（不同 item 被编码成同一串 codes），必要时对碰撞组启用 Sinkhorn-Knopp 做迭代重编码（最多 20 轮）。

### 2.4 重要限制（务必看）

`index/generate_indices.py` 当前**直接用行号 `0..N-1` 作为 item_id key**，不读取 `*.ids.json`。

这意味着：

- 如果你的数据集内部的 item_id 就是 `0..N-1` 且与 `.inter.json/.item.json` 完全一致：没问题；
- 如果你需要保留原始 item_id：建议用下方的 `tokenizer/` 方式（支持 `--ids_path`），或自行改造 `generate_indices.py` 的 key 写入逻辑。

---

## 3. 方式 B：`tokenizer/`（Residual K-Means / OpenOneRec-style）

入口与脚本：

- 训练 tokenizer：`tokenizer/train_res_kmeans.py`
- 导出 index json：`tokenizer/build_index_json.py`（支持 `--ids_path`）
- 一键脚本（Amazon 多数据集 shared tokenizer）：`tokenizer/amazon_train_and_export_index.sh`
- 详细说明：`tokenizer/README.md`

### 3.1 训练 tokenizer（支持多 `.npy` 合并）

```bash
python3 tokenizer/train_res_kmeans.py \
  --data_paths \
    ./data/Arts/Arts.emb-qwen3-embedding-4B-td.npy \
    ./data/Automotive/Automotive.emb-qwen3-embedding-4B-td.npy \
  --model_path ./data/_shared_tokenizer/reskmeans_qwen3-embedding-4B \
  --n_layers 3 \
  --codebook_size 8192 \
  --dim 4096 \
  --niter 20 \
  --max_train_points 200000
```

输出默认是：`<model_path>/model.pt`（包含每层 codebook）。

### 3.2 导出每个数据集的 `Dataset.index_*.json`（可保留原始 item_id）

```bash
python3 tokenizer/build_index_json.py \
  --model_path  ./data/_shared_tokenizer/reskmeans_qwen3-embedding-4B/model.pt \
  --emb_path    ./data/Arts/Arts.emb-qwen3-embedding-4B-td.npy \
  --ids_path    ./data/Arts/Arts.emb-qwen3-embedding-4B-td.ids.json \
  --output_path ./data/Arts/Arts.index_qwen3-embedding-4B.json \
  --device cuda \
  --batch_size 10000
```

如果不提供 `--ids_path`，脚本会退化为用 `0..N-1` 做 item_id（不推荐）。

---

## 4. 两种方式怎么选？

| 维度 | `index/`（RQVAE） | `tokenizer/`（Residual KMeans） |
|---|---|---|
| 方法 | 深度模型学习非线性映射 | 逐层 KMeans residual 量化 |
| 训练成本 | 高（PyTorch 训练，建议 GPU） | 低（Faiss KMeans，可 CPU/单卡 GPU） |
| 多数据集共享 codebook | 不自然（需合并训练且仍要处理 id） | 天然支持（一个 tokenizer，多数据集导出） |
| item_id 映射 | 默认用行号 `0..N-1` | 支持 `--ids_path` 保留原始 id |
| 适用场景 | 追求更强表达能力/更低碰撞 | 快速、稳定、可复用 tokenizer |

---

## 5. 下游使用（SFT / RL / Test）

索引生成后，在训练/评测阶段统一通过 `--index_file` 指定：

```bash
--data_path ./data \
--dataset Instruments \
--index_file .index_qwen3-embedding-4B.json
```

请确保文件存在：`./data/Instruments/Instruments.index_qwen3-embedding-4B.json`。

---

## 6. 常见问题（FAQ）

1. **报错找不到 index 文件**
   - 多半是 `--index_file` 拼接规则没对上：确认文件名是 `{dataset}{index_file}`，且在 `{data_path}/{dataset}/` 下。
2. **训练时新增 token 数量不对 / 推理时 tokenizer 大小不一致**
   - 训练会从 `index_file` 里收集 `<a_*>` 等 token 并 `tokenizer.add_tokens(...)`；推理加载 LoRA 时也会根据 `adapter_config.json` 处理词表扩展。请确保 `index_file` 与训练时一致。
3. **`index/` 导出的 key 和 `.item.json`/`.inter.json` 对不上**
   - 这是 `generate_indices.py` 用行号写 key 导致的；如果你的 item_id 不是 `0..N-1`，建议改用 `tokenizer/build_index_json.py` 或自行改造生成逻辑。
