# 测试 / 评测说明

评测相关代码主要在 `src/seqrec/`（序列推荐）与 `src/text_generation/`（文本生成）。可直接运行 Python 模块，也可以用 `scripts/` 下的脚本模板快速启动。

> `scripts/*/*.sh` 多为“可运行模板”：首次使用请先改 `DATASET/DATA_PATH/CKPT_PATH/BASE_MODEL/INDEX_FILE` 等变量为你的实际路径。

---

## 1. 序列推荐（seqrec / fusionseqrec / item2index 等）

常用脚本：

- Case（看生成样例）：`scripts/seqrec/case_seqrec.sh`、`scripts/seqrec/case_item2index.sh`
- Metric（跑指标）：`scripts/seqrec/metric_ddp.sh`（多卡）、`scripts/seqrec/metric_seqrec.sh`（单卡/调试）
- 约束解码版本：`scripts/seqrec/metric_constrained_ddp.sh`、`scripts/seqrec/metric_constrained_seqrec.sh`

### 1.1 LoRA / 非 LoRA 的加载规则

序列评测入口一般是 `torchrun -m src.seqrec.metric_ddp ...`，关键参数：

- 非 LoRA：只需要 `--ckpt_path <checkpoint-*>`
- LoRA：需要同时提供：
  - `--ckpt_path <adapter 或 checkpoint 路径>`
  - `--base_model <base 模型路径>`
  - `--lora`

以 `scripts/seqrec/metric_ddp.sh` 为准（脚本里已包含 `--results_file` 的保存逻辑）。

### 1.2 用哪些参数切换任务？

- `--test_task`：控制评测任务（默认 `seqrec`；也可用 `item2index`/`fusionseqrec` 等）
- `--test_prompt_ids`：控制使用哪组 prompt（`all` 或逗号分隔的 id 列表）
- `--index_file`：加载 SID 索引（文件拼接规则见 `docs/sid_readme.md`）

### 1.3 `--model_type` 与 collator 路由

评测阶段会根据 `--model_type` 自动选择 collator：

- 文本自回归模型（`qwen2` / `qwen2_5` / `qwen2_instruct` / `qwen2_5_instruct` / `llama`）：走文本 collator 路径
- 多模态模型（`qwen2_vl` / `qwen2_5_vl` / `llava_onevision`）：走多模态 collator 路径

建议与训练时 `--model_type` 保持一致，避免 chat template 或 tokenizer 路由不匹配。

### 1.4 多数据集评测（重要更新）

`--dataset` 现在支持传入逗号分隔的多个数据集，并统一合并评测：

```bash
--dataset Arts,Games,Toys
```

更新后的行为：

- 多数据集会被合并成一个测试集对象（内部保持 `set_prompt` / `get_all_items` 等接口兼容）；
- 指标会在合并后的样本集合上统计；
- 适用于 `metric.py` / `metric_ddp.py` / `metric_constrained*.py` 主路径。

如果你想看“每个数据集单独指标”，建议分别单独运行三次，而不是一次性合并。

### 1.5 复现与性能取舍

所有主评测入口都支持全局参数：

- `--seed`
- `--deterministic`

建议：

- 日常跑分：不加 `--deterministic`（速度更快）
- 严格对比实验：加 `--deterministic`（结果更稳定，但速度更慢）

---

## 2. 文本生成（text_enrich 等）

常用脚本在 `scripts/text_generate/`：

- LoRA：`scripts/text_generate/evaluate_lora.sh`
- 非 LoRA：`scripts/text_generate/evaluate_*.sh`（按模型类型区分）

文本生成评测实现见 `src/text_generation/evaluate.py`，指标/输出文件路径以脚本参数为准。

---

## 3. 常见问题（FAQ）

1. **LoRA 评测报错找不到 base model**
   - 确认同时传了 `--lora --base_model --ckpt_path`。
2. **多数据集评测结果看起来和单数据集不一致**
   - 合并评测本质是“样本池混合统计”；若要可比，请按数据集分别跑。
3. **相同命令重复跑分有轻微波动**
   - 可加 `--deterministic`，换取更高一致性。
