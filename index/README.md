# GRec Index（RQVAE / SID 架构说明）

本目录实现 GRec 的 **SID（Semantic/Structured Indexing / 离散索引）**：把连续的 item embedding（`.npy`）编码成离散 token 序列，
并导出每个数据集目录下的 `Dataset.index_*.json`，作为下游 SFT/RL/评测阶段的 `--index_file`。

> 与 `../tokenizer/` 的区别：`index/` 用 **深度网络（RQVAE）** 学一个“embedding→code”的非线性映射；`tokenizer/` 用 **纯 KMeans 的 residual quantization**，更轻、更接近 OpenOneRec 官方 tokenizer。

---

## 1. 输入 / 输出

### 输入
- 每个 item 一行 embedding 的 `.npy`（形状约为 `[num_items, dim]`），例如：`./data/Arts/Arts.emb-qwen3-embedding-4B-td.npy`

### 输出
- `Dataset.index_{MODEL_NAME}.json`，格式：

```json
{
  "0": ["<a_12>", "<b_7>", "<c_1024>"],
  "1": ["<a_88>", "<b_3>", "<c_2047>"]
}
```

> 下游加载逻辑是：`data_path/dataset/(dataset + index_file)`  
> 所以你传 `--index_file .index_qwen3-embedding-4B.json` 时，文件应为 `Arts/Arts.index_qwen3-embedding-4B.json`。

---

## 2. 核心模型：RQVAE（Encoder → Residual VQ → Decoder）

对应代码：
- `models/rqvae.py`: `RQVAE`
- `models/rq.py`: `ResidualVectorQuantizer`
- `models/vq.py`: `VectorQuantizer`
- `models/layers.py`: `MLPLayers` + `sinkhorn_algorithm` + sklearn `kmeans`

整体结构：

1) **Encoder（MLP）**：把原始 embedding `x ∈ R^D` 映射到低维 latent `z ∈ R^{e_dim}`  
2) **Residual Vector Quantizer（RQ）**：多层量化器逐层量化 residual，得到离散 indices（每层一个 code）  
3) **Decoder（MLP）**：把量化后的 latent `z_q` 重构回 `x_hat ∈ R^D`  

训练目标（`models/rqvae.py::compute_loss`）：
- `reconstruction loss`：`mse(x_hat, x)` 或 `l1`
- `quantization loss`：来自 `VectorQuantizer` 的 codebook/commitment loss

最终优化的是：
`loss_total = loss_recon + quant_loss_weight * loss_quant`

---

## 3. 量化细节：最近邻 / Sinkhorn-Knopp（用于缓解碰撞）

`models/vq.py::VectorQuantizer.forward` 的关键：
- 默认用 **欧式距离最近邻**：`argmin ||z - e_k||^2`
- 当 `use_sk=True` 且 `sk_epsilon>0` 时，用 `sinkhorn_algorithm` 在 batch 内做一次近似的“平衡分配”，然后 `argmax(Q)` 取分配结果

这会影响两件事：
- **码本利用率**（每层用了多少个 code）
- **碰撞率**（不同 item 是否会被编码成同一串 codes）

---

## 4. 训练流程（main.py + trainer.py）

入口：`index/main.py`

数据：
- 单数据集：`datasets.py::EmbDataset(data_path)`
- 多数据集联合训练：`datasets.py::MultiEmbDataset(data_paths)`（把多个 `.npy` 当成一个大 dataset）

训练器：`trainer.py::Trainer.fit`
- 可选 **KMeans 初始化**：当 `--kmeans_init true --large_scale_kmeans true` 时，会先取最多 `20000` 个样本跑一次 `self.model(init_data)`，触发每层 VQ 的 sklearn KMeans 初始化（见 `trainer.py`）
- 每 `eval_step` 做一次评估：计算碰撞率 + 码本利用率（见 `trainer.py::_valid_epoch`）
- 会保存：
  - `best_loss_model.pth`
  - `best_collision_model.pth`
  - `best_utilization_model.pth`

> 注意：这里的 KMeans 来自 `sklearn.cluster.KMeans`（`models/layers.py::kmeans`），是 CPU 算法，不是 Faiss。

---

## 5. 生成索引文件（generate_indices.py）

入口：`index/generate_indices.py`

逻辑概览：
1) 加载 checkpoint（包含 `args` 和 `state_dict`）
2) 对全量 embedding 做 `model.get_indices(...)` 得到每个 item 的离散 indices
3) 把 indices 转成 token list（`<a_{}>`, `<b_{}>`, ...）
4) 若碰撞严重，脚本会对碰撞 item 组用 `use_sk=True` 迭代重编码（最多 20 轮）
5) 导出 `json`：`{item_id: [token,...]}`

**重要限制/坑：**
- `generate_indices.py` 目前把 key 写成 `0..N-1`（按行号枚举）。如果你的数据集 item_id 不是从 0 连续编号，需要你在生成阶段提供映射（例如你 embedding 旁边保存了 `*.ids.json`）。
- token 前缀目前写死到 `a..e`（最多 5 层）；如果你把 `num_emb_list` 设到 >5 层，需要扩展前缀表。

---

## 6. 计算/并行建议

- **是否需要 GPU**：RQVAE 是 PyTorch 模型，训练与推理建议用 GPU；CPU 也能跑但会慢很多。
- **是否支持多卡并行训练**：当前 `index/` 没有接入 DDP/FSDP（没有 `torchrun`/`DistributedDataParallel` 相关逻辑），默认是单进程单卡。
  - 如果你要多卡：通常的改法是把 `Trainer` 和 `DataLoader` 改成 DDP + DistributedSampler（本目录目前未实现）。

---

## 7. 和 `../tokenizer/` 的选择建议

选 `index/`（RQVAE）：
- 想让模型学到更强的非线性“embedding→codes”映射（可能更低的重构误差/更好的碰撞率）
- 可以接受更重的训练成本（GPU、长 epoch）

选 `tokenizer/`（Residual KMeans / OpenOneRec-style）：
- 想要更轻、更稳定、训练更快的离散化（基本就是 residual quantization）
- 想要在多个数据集上训练一个共享 tokenizer（shared codebook space）

