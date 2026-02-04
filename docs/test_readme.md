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

---

## 2. 文本生成（text_enrich 等）

常用脚本在 `scripts/text_generate/`：

- LoRA：`scripts/text_generate/evaluate_lora.sh`
- 非 LoRA：`scripts/text_generate/evaluate_*.sh`（按模型类型区分）

文本生成评测实现见 `src/text_generation/evaluate.py`，指标/输出文件路径以脚本参数为准。

