# OpenOneRec `pretrain/` vs GRec `finetune/train_ddp.py` 差异说明

本文对比两套训练系统的**数据如何加载、batch 如何组织、loss/mask 如何构造、分布式/训练循环/保存恢复**等关键差异，帮助你在需要“从预训练到微调/或反向迁移数据”时快速对齐概念。

对比对象：

- OpenOneRec 预训练：`src/OpenOneRec/pretrain/`（核心入口 `recipes/train_qwen3.py`）
- GRec 微调：`src/finetune/train_ddp.py`

> 说明：两边都基于 causal LM，但**训练目标与数据组织方式**差异很大：OpenOneRec 更偏“foundation model 的长上下文预训练/共训”，GRec 更偏“下游任务 SFT（prompt+answer）微调”。

---

## 1. 一句话总览（先建立直觉）

- **OpenOneRec pretrain**：以 **Parquet 流式数据**（`segments/messages`）为输入，`IterableDataset` 在线读取 + 本地 shuffle + **packing** 到超长序列（如 32k tokens），在 **FSDP** 下做大规模训练与 checkpoint（非 HF 直接格式，需要转换）。
- **GRec finetune/train_ddp.py**：以项目内自定义数据结构（JSON/目录 + 多任务 Dataset 类）为输入，构建 `ConcatDataset`，用 HF `transformers.Trainer`（DDP/DeepSpeed）做典型 SFT 微调（prompt → response），直接输出 HF 模型权重。

---

## 2. 启动方式与分布式初始化

### 2.1 OpenOneRec：MPI + FSDP

入口脚本通常是：

- `src/OpenOneRec/pretrain/examples/pretrain_stg1.sh`
- `src/OpenOneRec/pretrain/examples/pretrain_stg2.sh`
- `src/OpenOneRec/pretrain/examples/posttrain_sft.sh`

特征：

- 通过 `mpirun ... python3 recipes/train_qwen3.py ...` 启动多进程（多机/多卡）。
- rank/size 相关环境变量来自 OpenMPI：
  - `OMPI_COMM_WORLD_RANK`
  - `OMPI_COMM_WORLD_SIZE`
  - `OMPI_COMM_WORLD_LOCAL_RANK`
  - 初始化逻辑：`src/OpenOneRec/pretrain/recipes/train_qwen3.py` 中 `initialize_distributed()`
- 模型并行策略：
  - 使用 `shard_model(...)` 做 FSDP 参数分片（见 `initialize_model()`）
  - 使用 `device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,))` 建 DP mesh

### 2.2 GRec：torchrun/DDP 语义 + Transformers Trainer（可接 DeepSpeed）

入口脚本是：

- `src/finetune/train_ddp.py`

特征：

- rank/size 相关环境变量采用 torchrun/DDP 常见形式：
  - `LOCAL_RANK`
  - `WORLD_SIZE`
  - 见 `UnifiedTrainer.__init__` / `_setup_environment`
- 分布式由 `transformers.Trainer` 接管；配置通过 `TrainingArguments` 传入：
  - `ddp_find_unused_parameters=False if self.ddp else None`
  - `deepspeed=self.args.deepspeed`（`src/parser.py` 默认给了一个 ds config 路径）

**关键差异：**

- OpenOneRec：训练框架“自己写循环 + 自己做 FSDP + 自己做 checkpoint + 自己做 dataloader state”。
- GRec：训练框架主要“交给 HF Trainer/DeepSpeed”，脚本更多是组织数据/参数/日志。

---

## 3. 模型与词表（token 扩展）的核心差异

### 3.1 OpenOneRec：先扩 Qwen3 词表加入 itemic tokens（SID）

OpenOneRec 依赖 **itemic tokens**（`<s_a_i>`、`<s_b_i>`、`<s_c_i>`、`<|sid_begin|>`、`<|sid_end|>`）表达离散化的 item id（SID）：

- 词表扩展工具：`src/OpenOneRec/pretrain/tools/model_converter/expand_qwen3_vocab.py`
- 脚本封装：`src/OpenOneRec/pretrain/scripts/expand_qwen3_vocab.sh`
- 典型训练分两阶段：
  - Stage1：冻结 LLM，仅优化 embedding（通过 `--start_optimize_embedding_index` + `EmbeddingGradientMasker`）
  - Stage2：全参共训（rec data + general text）

训练脚本里对 embedding 冻结/解冻采用“梯度 mask + 恢复冻结参数”的方式（见：
`src/OpenOneRec/pretrain/recipes/train_qwen3.py` 的 `EmbeddingGradientMasker` 与 `restore_frozen_params()`）。

### 3.2 GRec：从数据集中收集新 token，resize embedding 后直接 Trainer 微调

GRec 的 token 扩展在 `load_model_for_training()` 里完成：

- 位置：`src/utils.py`
- `_extend_vocabulary()`：
  - 若 `new_tokens` 为空，会先 `load_datasets(...)`，再取 `train_data.datasets[0].get_new_tokens()`
  - `tokenizer.add_tokens(new_tokens)` + `model.resize_token_embeddings(new_vocab_size)`
  - 同时写入 `output_dir/token_meta.json`
- `train_ddp.py` 额外会保存 processor 与更新后的 config（递归更新 `vocab_size`），见：
  - `src/finetune/train_ddp.py` 的 `_save_configs()`

**关键差异：**

- OpenOneRec：token 扩展是“固定规则的 itemic token 集合”（按层/码本大小生成），更像基础设施；并且 Stage1 明确“只训新 embedding”。
- GRec：token 扩展是“依赖数据集 index 文件里的 token 集合”，更像“为某个数据集/索引定制的增量词表”；LoRA/Full 训练由参数控制。

---

## 4. 数据格式与数据准备：Parquet（OpenOneRec） vs 项目内 Dataset（GRec）

### 4.1 OpenOneRec：统一 Parquet 行格式（segments/messages）

格式规范在：

- `src/OpenOneRec/data/README.md`

OpenOneRec 的 dataloader 最终只关心每行至少包含：

- `uuid`（唯一）
- `source`（来源名，用于 datasource 监控）
- `segments`（预训练文本）**或** `messages`（对话数据）

对应解析代码在：

- `src/OpenOneRec/pretrain/onerec_llm/data/qwen3_dataset.py`
  - `Qwen3NaiveParquetDataset._parser()`：把 parquet 行转换成 `sample["json"]`
  - `Qwen3ChatCompletionDataset._process_completion()`：处理 `segments`
  - `Qwen3ChatCompletionDataset._process_chat()`：处理 `messages`（用 `tokenizer.apply_chat_template`）

数据准备通常通过 `src/OpenOneRec/data/` 下的脚本完成，例如：

- `src/OpenOneRec/data/scripts/split_data.py`：合并 + 按行切 shard + 生成 `file_list.json`
- `src/OpenOneRec/data/prepare_pretrain.sh`：把通用语料（general_text）与推荐语料（rec_data）混合后 split

### 4.2 GRec：任务化 Dataset（TrainingSample），底层文件多为 JSON

GRec 的数据加载不是“统一 parquet schema”，而是“多个任务各自 Dataset 类读项目约定的数据文件”：

- 数据加载入口：`src/utils.py` 的 `load_datasets()`
- Dataset 类示例：`src/data.py`（如 `SeqRecDataset`）
  - 常见依赖文件（按 dataset 目录组织）：
    - `<Dataset>.inter.json`（用户交互序列）
    - `<Dataset><index_file>`（item → token 序列映射）
- 每个 `__getitem__` 返回统一结构 `TrainingSample`：
  - 定义：`src/type.py`（`input_text`、`label_text`、可选 `image_path`）

`load_datasets()` 会按：

- `args.tasks`（逗号分隔任务）
- `args.dataset`（可逗号分隔多个数据集）

构建多个 Dataset，并用 `ConcatDataset` 拼成一个训练集/验证集：

- `train_data = ConcatDataset(train_datasets)`
- `valid_data = ConcatDataset(valid_datasets)`

**关键差异：**

- OpenOneRec：数据 schema 统一，靠 `source/segments/messages` 跑通全流程；更适合大规模混合语料。
- GRec：数据 schema 分散在各 task 的 Dataset 实现；更适合“任务可控/可解释”的下游微调。

---

## 5. 数据如何“被读进来”：文件粒度分发与 shuffle 策略

### 5.1 OpenOneRec：文件列表广播 + worker 按文件切分 + 本地 shuffle buffer

关键链路：

1) `recipes/train_qwen3.py` 读取 `--dataset_config`，并调用：
   - `onerec_llm/data/dataloaders.py:get_dataloader()`
2) 默认 `name="chat_completion_parquet"`：
   - 创建 `Qwen3ChatCompletionParquetDataset`
3) `Qwen3ChatCompletionParquetDataset._build_source_dataset()`：
   - 如果 `sources` 是 `*.json`：读取 JSON **文件路径列表**（通常就是 `file_list.json`）
   - rank0 做 `sort + shuffle`，并对每个 epoch 重复一次
   - `dist.broadcast_object_list` 把文件列表广播到所有 rank
4) `Qwen3NaiveParquetDataset.__iter__local_shuffle()`：
   - 全局 worker id = `rank*num_workers + worker`
   - `idx % total_num_workers == local_worker_idx` 选择自己负责的文件子集（文件粒度切分）
   - 逐文件读 parquet（row group 0），逐行放入 `LocalShuffleBuffer`（样本粒度打散）

> 这套机制的特点：不会为每个 epoch 重新构建所有样本的随机采样器，而是“文件重排 + 本地 buffer 随机出队”，更适合海量数据流式训练。

### 5.2 GRec：ConcatDataset + Trainer/DataLoader 的采样器

关键链路：

1) `train_ddp.py` 中 `load_datasets(args, ...)` 得到 `ConcatDataset`
2) 交给 `transformers.Trainer`：
   - 内部会为 map-style dataset 构造 sampler（分布式情况下通常是 `DistributedSampler`）
   - 训练时以“样本索引”为单位进行 shuffle/分发
3) 每个 batch 的 tokenization 在 collator 中发生（见下一节）

**关键差异：**

- OpenOneRec：更像“数据流系统”，天然适配超大数据（按文件切分、样本 buffer shuffle、可恢复 dataloader state）。
- GRec：更像“经典监督学习训练”，按样本索引 shuffle，依赖 `__len__` 与随机采样器。

---

## 6. batch 组织方式：packing（OpenOneRec） vs 常规 padding batch（GRec）

### 6.1 OpenOneRec：packing 到 max_length（token-budget 训练）

在 `Qwen3ChatCompletionDataset.__iter__()` 中：

- 不断读取样本，累积到 `buffer`
- 当 `cur_length + sample_length >= self.max_length` 时，把多个样本 **拼成一个长序列**（packing）
- packing 后还会做额外 pad（对齐到 8 的倍数并 +64），并生成：
  - `input_ids`（形状通常是 `[1, L]`）
  - `loss_mask`（同形状）
  - `itemic_id_mask`
  - `position_ids`
  - `cu_seqlens` / `sample_idx`（用于区分 pack 内部的样本边界）

因此：

- OpenOneRec 的 dataloader `batch_size=1`（见 `StatefulDataLoader(..., batch_size=1, collate_fn=lambda x: x[0])`）
- “一个训练 step” ≈ “一个 packed 序列”，但其中可能包含多个原始样本

### 6.2 GRec：batch 内 padding +（可选）gradient accumulation

GRec 的 `Trainer` step 更接近常规 HF 训练：

- dataset 返回 `TrainingSample`（文本/多模态原始信息）
- collator 在 batch 维度对齐长度（`padding="longest"` 或 `padding=True`）
- 再由 `per_device_train_batch_size` 与 `gradient_accumulation_steps` 决定有效 batch

**关键差异：**

- OpenOneRec：以“每 step 消费多少 token”为核心（max_length packing），token 利用率高，适合长上下文预训。
- GRec：以“每 step 消费多少样本”为核心（per_device_batch_size），更符合 SFT 的交互式样本组织。

---

## 7. loss / mask 的构造方式：loss_mask（OpenOneRec） vs labels=-100（GRec）

### 7.1 OpenOneRec：先生成 `loss_mask`，再在训练脚本里组装 labels 并手动 shift

数据侧：

- `segments` 数据：默认对所有 token 计算 loss，但会 mask 掉最后的 EOS（见 `_process_completion`）
- `messages` 数据：通过模板定位 assistant span，仅对 assistant 计算 loss（见 `_process_chat` + `_get_assistant_mask`）

训练侧（`recipes/train_qwen3.py:compute_forward_backward`）：

- `labels = input_ids * loss_mask + ignore_index * (1 - loss_mask)`
- **手动 shift**：`labels = concat(labels[:,1:], pad)` 对齐 next-token prediction
- 使用自定义 `CrossEntropyLoss` 或 `ChunkedLossComputer` 计算 loss

### 7.2 GRec：collator 直接构造 `labels`，交给 HF CausalLM 自动 shift

GRec 的 collator 策略是：

- 先把 `full_text = input_text + label_text + eos` tokenize 得到 `input_ids`
- `labels = input_ids.clone()`
- 若 `only_train_response=True`：
  - 把 prompt 部分（以及 padding）置为 `-100`
  - 这样模型 forward 时只在 response token 上计 loss

实现位置：

- 纯文本：`src/collator.py:Collator`
- chat template（无图）：`src/collator.py:ChatTemplateCollator`
- 多模态：`src/collator.py:MultiModalCollator`

**关键差异：**

- OpenOneRec：loss 的“mask + shift”在训练脚本里显式实现，mask 张量与 packed 结构强绑定。
- GRec：loss 的“mask”由 labels=-100 完成，shift 由 HF 模型内部处理；更标准、更易与 HF 生态对齐。

---

## 8. 训练循环与中间过程（optimizer/scheduler/compile 等）

### 8.1 OpenOneRec：自定义训练循环（无 eval）

`recipes/train_qwen3.py` 主循环特征：

- `while True: batch = next(data_iter)` 直到数据耗尽
- 每 step：
  - `to_cuda(batch)`
  - forward/backward（可选 chunked loss）
  - grad clip（`clip_grad_by_value`）
  - optimizer.step / scheduler.step / zero_grad
  - 按 `logging_per_step` 打印与写 TensorBoard
  - 按 `save_checkpoint_per_step`（以及早期 step=20/200）保存 checkpoint
- **没有 eval_dataset / early stopping**
  - 预训练更看重长程训练与监控统计（token、source loss、MFU 等）

### 8.2 GRec：Transformers Trainer（有 eval + early stopping）

`src/finetune/train_ddp.py` 特征：

- 通过 `TrainingArguments` 配置训练：
  - `num_train_epochs`
  - `gradient_accumulation_steps`
  - `eval_strategy/save_strategy`（epoch 或 steps）
  - `save_only_model=True`
  - 默认带 `deepspeed` 配置
- 构建 `Trainer(...)` 后直接 `trainer.train(resume_from_checkpoint=...)`
- 带验证集：
  - `eval_dataset=valid_data`
  - `EarlyStoppingCallback(early_stopping_patience=3)`（监控 `eval_loss`）
- 训练后：
  - `trainer.save_state()` + `trainer.save_model(output_dir=...)`
  - 直接是 HF 可用目录结构
- 额外特性：
  - 若满足条件会 `torch.compile(model)`（见 `train_ddp.py`）

**关键差异：**

- OpenOneRec：训练循环、日志、checkpoint、（可选）chunked loss 都在一个脚本里“手写”，可控但需要更强工程约束。
- GRec：训练循环由 Trainer 统一管理，可快速复用 HF 能力（eval/early-stop/deepspeed/日志上报等）。

---

## 9. checkpoint 保存与恢复：分布式 checkpoint（OpenOneRec） vs HF/Trainer checkpoint（GRec）

### 9.1 OpenOneRec：分布式 checkpoint + 可选 dataloader 状态

- 保存逻辑：`save_model_checkpoint(...)`（`recipes/train_qwen3.py`）
  - 保存模型 shard、optimizer、lr_scheduler
  - 若传入 dataloader：还会保存 `dataloader.state_dict()` 到 `dataloader_ckpt/rank{rank}.pt`
- 恢复逻辑：
  - `--resume_from` 指向 checkpoint 目录
  - `--resume_from_tag` 指定 step tag（如 `global_step5000`）
  - `--resume_training_state` 才会恢复 optimizer/scheduler/dataloader
- **训练产物不是直接 HF**：
  - 需要 `src/OpenOneRec/pretrain/scripts/convert_checkpoint_to_hf.sh` 把某个 step 转为 HF 目录

### 9.2 GRec：Trainer 原生 checkpoint + 直接 save_model

- 恢复：
  - `trainer.train(resume_from_checkpoint=...)`
- 保存：
  - `trainer.save_state()` + `trainer.save_model()`
  - 外加脚本自己的 `processor.save_pretrained()` 与 `config.save_pretrained()`（`train_ddp.py:_save_configs`）

**关键差异：**

- OpenOneRec：面向大规模训练，checkpoint 更重（含 dataloader state），但需要额外“转换到 HF”步骤。
- GRec：面向快速微调与交付，产物直接 HF，可立即用于推理/继续训练。

---

## 10. “step/epoch”语义差异（很容易对齐错）

- OpenOneRec：
  - `global_step` = “产生了多少个 packed 序列”
  - 一个 packed 序列里可能有多个样本（`num_samples = len(cu_seqlens)-1`）
  - lr 的 `num_training_steps` 仅影响 scheduler 形状；训练是否停止取决于数据是否耗尽（以及 dataset_config 的 `num_epochs`）
- GRec：
  - step 由 Trainer 根据 `len(train_dataset)`、`per_device_batch_size`、`grad_accum`、`world_size` 推导
  - epoch 是 Trainer 的显式概念（`num_train_epochs`）
  - eval/save 的策略与 step/epoch 明确绑定

---

## 11. 可扩展性与“混合训练”的方式

### 11.1 OpenOneRec：通过数据文件本身混合（source 可监控）

想 mix 多域数据，一般做法是：

- 先把不同来源的 Parquet（统一 schema）合并/采样/分片
- 训练时 `source` 字段可用于监控每类数据的 loss/token 占比（脚本支持 `--monitor_datasource_loss/cnt`）

### 11.2 GRec：通过 tasks/dataset 参数组合 + 采样控制

混合训练常见入口：

- `--tasks`：决定要构造哪些 task 的 Dataset
- `--dataset`：可逗号分隔多个数据集目录
- `--train_prompt_sample_num` / `--train_data_sample_num`：控制每个 task 的 prompt 采样数与样本数

最终通过 `ConcatDataset` 合并，训练时“样本多的任务自然占比更大”（除非你显式下采样）。

---

## 12. 结论：何时用哪套（按目标选择）

- 你要做的是“**基础能力共训/长上下文预训练/大规模混合语料**”：
  - 选 OpenOneRec pretrain（Parquet 流式 + packing + FSDP + 强 checkpoint）
- 你要做的是“**在特定数据集/任务上做 SFT 微调（含多任务/多模态/LoRA）并快速交付 HF 权重**”：
  - 选 GRec `train_ddp.py`（Trainer 生态 + eval/early-stop + 直接 save_pretrained）

---

## 附：本文引用的关键文件索引

- OpenOneRec 训练入口：`src/OpenOneRec/pretrain/recipes/train_qwen3.py`
- OpenOneRec dataloader：`src/OpenOneRec/pretrain/onerec_llm/data/dataloaders.py`
- OpenOneRec parquet dataset：`src/OpenOneRec/pretrain/onerec_llm/data/qwen3_dataset.py`
- OpenOneRec 数据格式规范：`src/OpenOneRec/data/README.md`
- OpenOneRec split 脚本：`src/OpenOneRec/data/scripts/split_data.py`
- OpenOneRec vocab 扩展：`src/OpenOneRec/pretrain/tools/model_converter/expand_qwen3_vocab.py`
- GRec 训练入口：`src/finetune/train_ddp.py`
- GRec 数据加载：`src/utils.py`（`load_datasets`, `load_model_for_training`）
- GRec Dataset 示例：`src/data.py`
- GRec Collator：`src/collator.py`
- GRec 参数定义：`src/parser.py`
