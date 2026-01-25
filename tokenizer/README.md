# Residual K-Means Tokenizer (OpenOneRec-style Index)

这个目录提供一套 **OpenOneRec 风格**的 residual K-means tokenizer，用于把连续的 item embedding 量化成离散 codes，
并最终导出每个 dataset 目录下的 `*.json`（GRec/OpenOneRec 训练时的 `index_file`）。

> 如果你在看 GRec 的另一套 `index/`：它是 **RQVAE（深度模型）** 的离散化；本目录是 **纯 KMeans 的 residual quantization**（更轻、更接近 OpenOneRec 官方 tokenizer）。

核心目标（对应你的需求）：
- 从 `index/text2emb.py` 产出的 `.npy embedding` 出发
- **统一**在多个数据集上训练一个 tokenizer（shared index space）
- 为 `index/scripts/text2emb.sh` 里的所有数据集，分别导出 `Dataset.index_*.json`

## 架构：Residual K-Means Tokenizer 在做什么

一句话：它在做 **Residual Vector Quantization（RVQ）**，但每一层的 codebook 不是用梯度学的，而是用 KMeans 学的。

### 输入 / 输出

- 输入：`[N, D]` 的 item embedding（通常来自同一个 embedding model，例如 Qwen3-Embedding-4B）
- 输出：每个 item 一个长度为 `n_layers` 的 code 序列（每层一个整数），最终转成 token list：
  - 第 0 层：`<a_{code}>`
  - 第 1 层：`<b_{code}>`
  - ...

### 训练（train_res_kmeans.py）

训练过程是逐层的：
1) 第 0 层：对原始向量 `x` 做 KMeans，得到 centroid codebook `C0`
2) 编码：对每个向量选最近的 centroid `c0(x)`，并更新 residual：`r1 = x - c0(x)`
3) 第 1 层：对 residual `r1` 再做 KMeans 得到 `C1`
4) 重复直到 `n_layers`

这就是经典 RVQ / residual kmeans。

**关于 GPU / 多卡：**
- 这不是 PyTorch 的反向传播训练，所以没有 DDP“多卡训练”的概念。
- 训练主要依赖 Faiss KMeans（`faiss.Kmeans`）：
  - 默认 CPU（`faiss-cpu`）
  - 可选单卡 GPU（需要 `faiss-gpu`，训练时加 `--faiss_gpu` 或脚本里 `FAISS_GPU=1`）

### 推理与落盘（build_index_json.py / infer_res_kmeans.py）

本目录有两个“推理”脚本，输出格式不同：

- `build_index_json.py`：**.npy → GRec/OpenOneRec 可直接用的 index JSON**
  - 输入：`.npy` +（可选）`*.ids.json`
  - 输出：`Dataset.index_*.json`（token list）
  - 这是你现在 pipeline 的主入口

- `infer_res_kmeans.py`：**parquet → parquet（更偏 debug/分析）**
  - 输入：parquet(`pid`, `embedding`)
  - 输出：parquet(`pid`, `codes`)（codes 是整数数组）
  - 适合做离线分析/可视化/算重构误差，不直接满足 GRec 的 `index_file` 需求

## 目录文件

- `res_kmeans.py`: residual K-means 模型（Faiss KMeans）
- `train_res_kmeans.py`: 训练 tokenizer（支持多 `.npy` 合并训练）
- `build_index_json.py`: 用 tokenizer 对 `.npy` embedding 编码并导出 `*.index_*.json`
- `amazon_train_and_export_index.sh`: 一键脚本（多数据集训练 + 导出每个数据集 index json）

## 依赖安装

```bash
pip install torch numpy pyarrow faiss-cpu tqdm
```

如果你需要从 parquet 读取/写入（可选）：

```bash
pip install pandas
```

## 端到端 Pipeline（Amazon 类数据）

### 0) 你的数据目录结构（约定）

假设 `DATA_ROOT=/mnt/.../data`，每个数据集一个子目录：

```
$DATA_ROOT/Arts/
$DATA_ROOT/Automotive/
...
```

并且每个数据集目录里至少有（用于抽取 embedding）：

- `$DATA_ROOT/<DATASET>/<DATASET>.item.json`

### 1) 先抽取 embedding（amazon_text2emb）

脚本在 `src/GRec/index/scripts/text2emb.sh`，默认会为每个数据集生成：

- `$DATA_ROOT/<DATASET>/<DATASET>.emb-${PLM_NAME}-td.npy`
- `$DATA_ROOT/<DATASET>/<DATASET>.emb-${PLM_NAME}-td.ids.json`（用于保持 item_id 与 embedding 行对齐，强烈建议保留）

示例（在 `src/GRec/` 下执行）：

```bash
export HOME_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian
export PLM_NAME=qwen3-embedding-4B
bash index/scripts/text2emb.sh
```

### 2) 统一训练 tokenizer（shared index）

推荐直接用一键脚本（在本目录执行或用 `bash` 指定路径都行）：

```bash
DATA_ROOT=/mnt/.../data \
PLM_NAME=qwen3-embedding-4B \
MAX_TRAIN_POINTS=200000 \
bash amazon_train_and_export_index.sh
```

脚本已拆分为多个子步骤脚本，你也可以按需只跑其中一段：

```bash
# 只检查输入文件是否齐全
bash amazon_train_and_export_index.sh check

# 只训练 tokenizer（输出到 $TOKENIZER_OUT/model.pt）
bash amazon_train_and_export_index.sh train

# 只导出每个数据集的 *.index_*.json（需要已有 tokenizer）
MODEL_PT=/path/to/model.pt bash amazon_train_and_export_index.sh export
```

说明：
- `MAX_TRAIN_POINTS=0` 表示用全量训练（可能非常大）；多数据集合并时一般建议先采样训练。
- 输出 tokenizer 默认保存到：`$DATA_ROOT/_shared_tokenizer/reskmeans_${INDEX_NAME}_L${N_LAYERS}_C${CODEBOOK_SIZE}/model.pt`

如果你想手动训练（可控性更强）：

```bash
python3 train_res_kmeans.py \
  --data_paths \
    $DATA_ROOT/Arts/Arts.emb-qwen3-embedding-4B-td.npy \
    $DATA_ROOT/Automotive/Automotive.emb-qwen3-embedding-4B-td.npy \
    $DATA_ROOT/Cell/Cell.emb-qwen3-embedding-4B-td.npy \
    $DATA_ROOT/Games/Games.emb-qwen3-embedding-4B-td.npy \
    $DATA_ROOT/LC-Rec/LC-Rec.emb-qwen3-embedding-4B-td.npy \
    $DATA_ROOT/Pet/Pet.emb-qwen3-embedding-4B-td.npy \
    $DATA_ROOT/Sports/Sports.emb-qwen3-embedding-4B-td.npy \
    $DATA_ROOT/Tools/Tools.emb-qwen3-embedding-4B-td.npy \
    $DATA_ROOT/Toys/Toys.emb-qwen3-embedding-4B-td.npy \
  --model_path $DATA_ROOT/_shared_tokenizer/reskmeans_qwen3-embedding-4B \
  --n_layers 3 \
  --codebook_size 8192 \
  --dim 4096 \
  --niter 20 \
  --max_train_points 200000
```

### 3) 为每个数据集导出 index JSON（落到 dataset 目录下）

一键脚本会自动导出：

- `$DATA_ROOT/<DATASET>/<DATASET>.index_${INDEX_NAME}.json`

你也可以单独对某个数据集导出：

```bash
python3 build_index_json.py \
  --model_path $DATA_ROOT/_shared_tokenizer/reskmeans_qwen3-embedding-4B/model.pt \
  --emb_path   $DATA_ROOT/Arts/Arts.emb-qwen3-embedding-4B-td.npy \
  --ids_path   $DATA_ROOT/Arts/Arts.emb-qwen3-embedding-4B-td.ids.json \
  --output_path $DATA_ROOT/Arts/Arts.index_qwen3-embedding-4B.json \
  --device cuda \
  --batch_size 10000
```

导出的 JSON 结构为：

```json
{
  "0": ["<a_12>", "<b_7>", "<c_1024>"],
  "1": ["<a_88>", "<b_3>", "<c_2047>"]
}
```

其中 key 是 item_id（字符串），value 是 token list；下游会 `''.join(tokens)` 得到最终 item token 串。

## 在 GRec / OpenOneRec 训练中使用

假设你导出的是 `Instruments.index_qwen3-embedding-4B.json`，那么：

- `--data_path` 指向包含各 dataset 子目录的父目录（例如 `./data` 或 `$DATA_ROOT`）
- `--dataset Instruments`
- `--index_file .index_qwen3-embedding-4B.json`

因为代码会按如下规则读取索引：
`$DATA_ROOT/Instruments/Instruments${index_file}`

## 常见坑

- 训练 tokenizer 时如果直接用全量多数据集 embedding，可能非常占内存；优先用 `--max_train_points` 采样。
- 强烈建议保留 `text2emb.py` 导出的 `*.ids.json`，否则默认用 `0..N-1` 作为 item_id，可能和原始 item_id 不一致。

## 和 `../index/` 的核心区别（什么时候选哪个）

| 维度 | `tokenizer/`（本目录） | `index/`（RQVAE） |
|---|---|---|
| 方法 | residual KMeans / RVQ | Encoder-Quantizer-Decoder（深度模型） |
| 训练方式 | KMeans（无梯度） | PyTorch 训练（有梯度） |
| 训练成本 | 低；CPU 或单卡 GPU | 高；强烈建议 GPU |
| 多卡 | 不涉及 DDP | 当前实现默认单卡（未接 DDP） |
| 输出 | `Dataset.index_*.json`（token list） | `Dataset.index_*.json`（token list） |
| 适用场景 | 快速、稳定、共享 codebook | 更强表达能力，追求更低碰撞/更好重构 |

更完整的 `index/` 架构说明见：`../index/README.md`
