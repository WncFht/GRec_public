# SFT / Finetune（监督微调）说明

本项目的 SFT 阶段做两件事：

1. 让模型学会“推荐/检索/生成”这些下游任务的 **prompt → response** 映射；
2. 让模型把 SID 产出的 item 离散 token（例如 `<a_12><b_7><c_1024>`）当作**原子 token** 来理解/生成（通常需要扩词表）。

> 建议先完成 `docs/sid_readme.md` 的 SID 构建，确保 `data/<DATASET>/<DATASET>.index_*.json` 已就绪。

---

## 1. 训练入口与脚本选择

代码入口在 `src/finetune/`，常用脚本：

1. `src.finetune.train_ddp_vl.py`
   - 面向多模态 VLM（`qwen2_vl` / `qwen2_5_vl` / `llava_onevision`）
   - **会从 `index_file` 收集 `<a_*>` 等新 token 并扩展词表**
2. `src.finetune.train_ddp_vl_nonewtoken.py`
   - 与上面类似，但**不扩展词表**（仅适用于你确定 tokenizer 已包含这些 token 的场景）
   - 实验经验：如果 token 没有被当成原子 token，最后一个 token 可能更难学（需要结合具体 tokenizer 分词结果排查）
3. `src.finetune.train_ddp.py`
   - 面向纯文本 LLM（`qwen2` / `qwen2_5` / `qwen2_instruct` / `qwen2_5_instruct` / `llama` 等）
   - 同样会按需扩词表（从 `index_file` 收集）
4. `src.finetune.train_muon.py`
   - 使用 Muon 优化器的实验脚本（当前不是主线）

Shell 示例脚本在 `scripts/finetune/`（建议先看这里的“可运行模板”再改参数）。

当前 `parse_global_args` 支持的 `--model_type` 为：

- `qwen2_vl`
- `qwen2_5_vl`
- `llava_onevision`
- `qwen2`
- `qwen2_5`
- `qwen2_instruct`
- `qwen2_5_instruct`
- `llama`

---

## 2. 数据目录与必要文件

训练时 `--data_path` 指向数据根目录（默认 `./data`），`--dataset` 是数据集子目录名。

典型结构（以 `Instruments` 为例）：

```
data/Instruments/
  Instruments.index_*.json      # 必需：SID 离散索引（由 sid 阶段产出）
  Instruments.inter.json        # seqrec / fusionseqrec 等序列任务需要
  Instruments.item.json         # item2index / fusionseqrec 等需要（包含 title/description/图片字段等）
  images/                       # 多模态任务可能需要（由 --image_path 控制）
```

不同任务需要的文件不同（以 `src/data.py` 的实现为准）：

- `seqrec`：`<DATASET>.inter.json` + `<DATASET><index_file>`
- `fusionseqrec`：`<DATASET>.inter.json` + `<DATASET>.item.json` + `<DATASET><index_file>`
- `item2index` / `index2item`：`<DATASET>.item.json` + `<DATASET><index_file>`
  - 任务带 `_nosplit` 时：不做 8:1:1 item 划分，训练/验证/测试会在同一份 item 集合上构建（见 `ItemFeatDataset`）

---

## 3. 最小可跑示例（多卡 + LoRA + ZeRO2）

下面是一个典型的多卡训练命令（与 `scripts/finetune/train_qwen2-VL-7B.sh` 同源，参数可按需改）：

```bash
torchrun --nproc_per_node=4 --master_port=33325 -m src.finetune.train_ddp_vl \
  --seed 42 \
  --model_type qwen2_vl \
  --base_model ckpt/base_model/Qwen2-VL-7B-Instruct \
  --output_dir ./ckpt/Instruments/Qwen2-VL-7B-lora-item2index-seqrec-fusionseqrec \
  --data_path ./data \
  --dataset Instruments \
  --index_file .index_qwen3-embedding-4B.json \
  --tasks item2index,seqrec,fusionseqrec \
  --train_prompt_sample_num 1,1,1 \
  --train_data_sample_num 0,0,0 \
  --ratio_dataset 1 \
  --per_device_batch_size 12 \
  --gradient_accumulation_steps 2 \
  --use_gradient_checkpointing \
  --num_workers 32 \
  --learning_rate 5e-5 \
  --epochs 4 \
  --weight_decay 0.01 \
  --save_and_eval_strategy epoch \
  --deepspeed ./config/ds_z2_bf16.json \
  --bf16 \
  --use_lora \
  --lora_modules_to_save "embed_tokens,lm_head" \
  --only_train_response \
  --report_to wandb
```

有效 batch size 计算：

`effective_bs = nproc_per_node * per_device_batch_size * gradient_accumulation_steps`

上例为：`4 * 12 * 2 = 96`。

---

## 4. 关键参数解释（与代码一一对应）

参数解析集中在 `src/parser.py`：

### 4.1 全局参数

- `--model_type`：模型类型（必填），决定加载哪类模型/processor
  - 当前可选：`qwen2_vl`、`qwen2_5_vl`、`llava_onevision`、`qwen2`、`qwen2_5`、`qwen2_instruct`、`qwen2_5_instruct`、`llama`
- `--deterministic`：启用严格可复现模式（通常更慢）
  - 默认不传时为性能优先：启用 cuDNN，`benchmark=True`
  - 传入后会切换为 deterministic 算法与更保守的 cudnn 配置

### 4.2 数据相关参数（dataset_args）

- `--data_path`：数据根目录（默认 `./data`）
- `--dataset`：数据集名（可逗号分隔，表示多数据集拼接训练）
- `--index_file`：索引文件后缀（必填），例如 `.index_qwen3-embedding-4B.json`
- `--tasks`：训练任务列表，英文逗号分隔
- `--train_prompt_sample_num`：每个任务“每条数据采样多少种 prompt”（长度必须与 tasks 一致）
- `--train_data_sample_num`：每个任务“最多采样多少条数据”（`0` 表示全量；长度必须与 tasks 一致）
- `--ratio_dataset`：对 `*.inter.json` 的用户数做截断比例（便于快速实验）
- `--max_his_len` / `--his_sep` / `--add_prefix`：控制序列任务 history 的长度与格式
- `--only_train_response`：只在 response 部分计算 loss（更贴近指令微调习惯）

### 4.3 训练相关参数（train_args）

- `--per_device_batch_size`、`--gradient_accumulation_steps`、`--epochs`、`--learning_rate` 等常规训练参数
- `--use_lora`：启用 LoRA
- `--lora_target_modules`：LoRA 注入模块名列表（逗号分隔）
- `--lora_modules_to_save`：在 LoRA 模式下仍“全量保存/训练”的模块
  - 推荐至少包含 `embed_tokens,lm_head`，因为 SID 里的 `<a_*>` 等 token 是新增 token，其 embedding/输出头需要可训练且能保存
- `--freeze`：冻结策略（如只训练 embedding 等）
- `--deepspeed`：DeepSpeed 配置文件路径（项目自带 `config/ds_z2_*.json` 与 `config/ds_z3_*.json`）
- `--resume_from_checkpoint`：从某个 checkpoint 或 LoRA adapter 恢复（注意：训练脚本默认倾向 `save_only_model`，若你需要完整训练状态请自行调整）
- `--save_and_eval_strategy`：`epoch` / `steps` / `no`
  - 若任务组合无法构建 valid（例如全是 `*_nosplit`），请使用 `--save_and_eval_strategy no`
- `--eval_by_dataset`：按数据集分别构建验证集并单独记录指标（如 `eval_Arts_loss`）
- `--eval_main_dataset`：在 `--eval_by_dataset` 场景指定“用于 best model / early stop 的主验证集”

### 4.4 训练行为更新（重要）

- `torch.compile` 现在在创建 `Trainer` 之前执行，编译结果会真正被训练使用。
- 开启 `torch.compile` 后，前几个 step 可能有编译 warmup 开销，这是预期现象。
- 训练入口会根据 `--save_and_eval_strategy` 检查是否必须存在 valid split：
  - 需要评估却没有 valid，会直接报清晰错误；
  - 不需要评估（`no`）则允许 `valid_data=None`。

---

## 5. 产物、合并与推理/评测

### 5.1 训练产物在 `--output_dir`

训练会在 `output_dir` 下产生：

- `checkpoint-*/`：分步 checkpoint
- `processor/` 或 tokenizer/config：用于推理时保证分词一致
- `token_meta.json`：记录新增 token 数量等元信息（用于排查词表不一致问题）

### 5.2 ZeRO3 合并（可选）

如果你用的是 ZeRO3，checkpoint 可能是分片形式（依赖具体 deepspeed 配置），可参考：

- `convert/convert.sh`（调用 `convert/zero_to_fp32.py` 等脚本，把 ZeRO 分片转换为可加载的权重）
- LoRA 合并：`src/merge_lora.py`（把 adapter 合并进 base model，便于单模型推理）

### 5.3 推理/评测怎么加载

序列推荐评测脚本在 `scripts/seqrec/`，并在 `docs/test_readme.md` 有说明：

- **LoRA 推理**：通常需要同时提供 `--base_model` 与 `--ckpt_path`（adapter/checkpoint 路径），并传 `--lora`
- **全量模型推理**：通常只需提供 `--ckpt_path`

---

## 6. 常见问题（FAQ）

1. **`train_prompt_sample_num/train_data_sample_num` 报长度不匹配**
   - 这两个参数必须与 `--tasks` 一一对应，逗号分隔后的元素个数要相同（见 `src/utils.py::load_datasets` 的 assert）。
2. **`index_file` 找不到**
   - 检查拼接规则：文件名应为 `data/<DATASET>/<DATASET><index_file>`（例如 `Instruments.index_xxx.json`）。
3. **LoRA 训练后推理发现新增 token 不生效**
   - 确认训练时 `--lora_modules_to_save` 包含 `embed_tokens,lm_head`，并确保推理时加载到同一份 tokenizer/processor（训练脚本会保存到输出目录）。
4. **报错：`No validation datasets were built`**
   - 常见于任务全是 `*_nosplit`（没有 valid 划分）但仍启用了 eval。
   - 解决方案：
     - 若只是训练：加 `--save_and_eval_strategy no`；
     - 若需要 eval：在 `--tasks` 里加入可构建 valid 的任务（如 `seqrec` / `item2index` 非 nosplit）。
5. **同样配置每次结果不完全一致**
   - 加 `--deterministic` 提高可复现性；
   - 但会带来吞吐下降，建议只在对比实验或排障时开启。

---

## 7. 环境变量参数总表

SFT 相关脚本（`train_text.sh`、`bundle_train_common.sh`、`bundle_metric_common.sh`）的完整环境变量说明（含默认值）见：

- `docs/finetune_env_vars.md`
