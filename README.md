# GRec 项目总览

GRec 聚焦**多模态生成式推荐**。整体流水线是：

**Embedding → SID（离散索引）→ SFT（监督微调）→（可选）RL（排序优化）→ Test（评测/回归）**

文档入口：

- 环境安装：`docs/install.md`
- SID：`docs/sid_readme.md`（包含两种 SID 构建方式：`index/` 与 `tokenizer/`）
- SFT：`docs/finetune_readme.md`
- RL：`docs/rl_readme.md`
- 测试：`docs/test_readme.md`
- 数据处理：`docs/dataprocess_readme.md`
- Notebook：`docs/notebook_readme.md`
- 集群脚本与命名规范：`docs/run_script_convention.md`

---

## 项目结构（你需要知道的目录）

- `data_process/`：数据增强、图片下载、embedding 抽取等工具
- `index/`：SID 方式 A（RQVAE，深度离散化）+ `text2emb.py`（文本 embedding 抽取）
- `tokenizer/`：SID 方式 B（Residual KMeans，OpenOneRec-style tokenizer）
- `src/`：训练/评测核心代码（SFT、RL、SeqRec metric、Text generation 等）
- `scripts/`：一键/模板脚本（finetune、seqrec、text_generate、rl…）
  - 统一模板入口说明：`scripts/finetune/README.md`、`scripts/rl/README.md`
- `config/`：deepspeed/accelerate 配置与 benchmark 配置

---

## 环境准备（简版）

- Python：建议 `3.10`（训练/flash-attn/deepspeed 兼容性更稳，详见 `docs/install.md`）
- 依赖：`pip install -r requirements.txt`
- RL 额外依赖（若你要跑 `docs/rl_readme.md`）：`pip install trl bitsandbytes`
- DeepSpeed 配置：`config/ds_z2_*.json` / `config/ds_z3_*.json`
- accelerate 配置（RL 常用）：`config/zero2_opt.yaml`

> `setup.sh` / `Dockerfile` 提供了环境搭建参考，但可能需要按你的 CUDA/Python 版本做调整。

---

## 快速开始：从 0 跑通一条链路

### 1) 准备 embedding（可选但常用）

- 纯文本 embedding：`index/scripts/text2emb.sh`（读取 `data/<DATASET>/<DATASET>.item.json`，产出 `*.emb-*.npy` + `*.ids.json`）
- 多模态 embedding：`scripts/extract_rep.py`（包装 `data_process/qwen_embeddings.py`，可批量跑多种 mode）

### 2) SID：生成离散索引（两种方式）

现在 SID 有两套实现（建议先看 `docs/sid_readme.md`）：

- **方式 A：`index/`（RQVAE）**：`index/main.py` 训练 → `index/generate_indices.py` 导出 `Dataset.index_*.json`
- **方式 B：`tokenizer/`（Residual KMeans）**：`tokenizer/train_res_kmeans.py` 训练 tokenizer → `tokenizer/build_index_json.py` 导出 `Dataset.index_*.json`

导出的索引文件最终通过 `--index_file` 在训练/评测阶段加载（拼接规则见 `docs/sid_readme.md`）。

### 3) SFT：多任务监督微调

详见 `docs/finetune_readme.md` 与 `scripts/finetune/`。

核心点：

- `train_ddp_vl.py` / `train_ddp.py` 会从 `index_file` 收集 `<a_*>` 等 token 并扩词表（推荐主线）
- LoRA 场景建议 `--lora_modules_to_save "embed_tokens,lm_head"`，否则新增 token 的 embedding/head 可能无法正确保存
- 支持 `--deterministic`：需要严格复现时开启；默认不启用时更偏性能

### 4)（可选）RL：排序优化

详见 `docs/rl_readme.md` 与 `scripts/rl/`。

当前 RL 入口为 `python -m src.rl.rl`（脚本模板见 `scripts/rl/`）。

### 5) Test：序列推荐与文本生成评测

详见 `docs/test_readme.md`，常用脚本：

- 序列推荐：`scripts/seqrec/case_seqrec.sh`、`scripts/seqrec/metric_ddp.sh`
- 文本生成：`scripts/text_generate/evaluate*.sh`、`scripts/text_generate/evaluate_lora.sh`

补充：

- `--dataset` 在评测阶段支持逗号分隔多数据集（会合并后统一统计指标）
- 若希望看每个数据集单独指标，建议分别运行多次

---

## 备注

- SID 输出文件的命名与 `--index_file` 的拼接规则是全流程最常见的踩坑点；强烈建议先通读 `docs/sid_readme.md`。
