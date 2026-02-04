# RL（强化学习/排序优化）阶段说明

GRec 的 RL 阶段用于在 SFT 之后，进一步用“规则/排序”类 reward 优化推荐生成质量（典型是让模型更稳定地产生**合法 item token 串**并提升命中）。

核心特点：

- 算法侧：基于 TRL 的 `GRPOConfig`（Group-style 优化），一次为同一 prompt 生成 `num_generations` 个候选，再做组内优势计算与更新。
- 生成侧：支持 **beam search** 或 sampling，并可启用“前缀约束”把输出限制在合法 item token 序列空间。
- reward 侧：内置 `format/rule/ndcg`，也支持通过 `--reward_funcs/--reward_weights` 自定义组合。

---

## 1. 代码入口与组成

RL 主入口：

- `src/rl/rl.py`：训练入口（参数解析、数据集构建、训练循环、保存与评估）

相关组件：

- `src/data_rl.py`：RL 专用 Dataset（把 `seqrec/fusionseqrec` 转成“Verl-style records”，并提供 ground_truth token ids）
- `src/rl/reward_fns.py`：reward 实现（`format_reward` / `rule_reward` / `ndcg_rule_reward`）
- `src/rl/minionerec_trainer.py`：自定义 Trainer（封装 GRPO 训练逻辑与生成/评估）
- `src/rl/LogitProcessor.py`：前缀约束用的 `ConstrainedLogitsProcessor`（避免生成非法 token）

脚本示例：

- `scripts/rl/*.sh`：accelerate/单卡启动模板（默认入口为 `src.rl.rl`）

---

## 2. 训练前置条件（必须满足）

### 2.1 SFT checkpoint

RL 训练一般从一个已经做过 SFT 的 checkpoint 开始：

- `--base_model`：可以是 HF 模型、也可以是本项目 SFT 输出的 `checkpoint-*` 目录

### 2.2 SID index 文件

RL 与 SFT 一样依赖 SID：

- 需要 `data/<DATASET>/<DATASET>.index_*.json`
- 通过 `--index_file .index_xxx.json` 指定（拼接规则见 `docs/sid_readme.md`）

### 2.3 数据文件

目前 `src/rl/rl.py` 只显式支持：

- `seqrec`（需要 `<DATASET>.inter.json` + `<DATASET><index_file>`）
- `fusionseqrec`（需要 `<DATASET>.inter.json` + `<DATASET>.item.json` + `<DATASET><index_file>`）

> 请务必显式传 `--tasks seqrec` 或 `--tasks fusionseqrec`。默认的 tasks 列表很长，RL 入口不会为未实现任务构建数据集。

### 2.4 依赖

RL 代码会用到：

- `trl`（提供 `GRPOConfig`）
- `accelerate`
- `bitsandbytes`（默认 optimizer 配置为 `paged_adamw_32bit`）

项目的 `requirements.txt` 可能未覆盖全部 RL 依赖时，请按环境补齐。

---

## 3. 快速启动（推荐用 accelerate）

典型启动方式参考 `scripts/rl/train_seqrec_ranking.sh`（这里给一个“最小可跑”的通用模板）：

```bash
accelerate launch \
  --config_file ./config/zero2_opt.yaml \
  --num_processes 4 --main_process_port 29503 \
  --module src.rl.rl \
  --model_type qwen2_5_instruct \
  --base_model ckpt/Instruments/<YOUR_SFT_CKPT> \
  --output_dir ckpt/Instruments/<YOUR_RL_OUT> \
  --data_path ./data \
  --dataset Instruments \
  --index_file .index_qwen3-embedding-4B.json \
  --tasks seqrec \
  --train_prompt_sample_num 1 \
  --train_data_sample_num 0 \
  --train_batch_size 64 \
  --eval_batch_size 128 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 16 \
  --beam_search \
  --max_completion_length 128 \
  --learning_rate 1e-5 \
  --beta 5e-3 \
  --reward_type ranking \
  --bf16
```

单卡 debug（不走 accelerate）也可以：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.rl.rl \
  --model_type qwen2_5_instruct \
  --base_model ckpt/Instruments/<YOUR_SFT_CKPT> \
  --output_dir ckpt/Instruments/<YOUR_RL_OUT> \
  --data_path ./data --dataset Instruments --index_file .index_xxx.json \
  --tasks seqrec --train_prompt_sample_num 1 --train_data_sample_num 0
```

---

## 4. 关键参数解释（按功能分组）

参数由 `src/parser.py::parse_rl_args/parse_dataset_args/parse_global_args` 定义。

### 4.1 基础（模型/数据）

- `--model_type`：模型类型（必填），影响 tokenizer/chat template
- `--base_model`：RL 的起点模型（通常是 SFT checkpoint）
- `--data_path`、`--dataset`、`--index_file`：数据与索引
- `--tasks`：目前只建议填 `seqrec` 或 `fusionseqrec`
- `--train_prompt_sample_num` / `--train_data_sample_num`
  - 这两个参数虽然 RL 数据集本身不依赖，但会影响 `load_model_for_training()` 内部的“扩词表”流程（会调用一次 SFT 的 `load_datasets()` 做 new tokens 收集）
  - 因此务必让它们与 `--tasks` 的长度一致（例如 `seqrec` 就传单个数：`1` / `0`）

### 4.2 生成（rollout）

- `--num_generations`：每个 prompt 生成多少个候选（组大小）
- `--beam_search`：开启 beam search（否则更偏 sampling）
- `--temperature`：采样温度
- `--max_completion_length`：completion 的最大 token 数

### 4.3 reward 组合

默认用 `--reward_type` 选择：

- `rule`：`format_reward + rule_reward`
- `ranking`：`format_reward + rule_reward + ndcg_rule_reward`
- `ranking_only`：`format_reward + ndcg_rule_reward`

也可以用 `--reward_funcs/--reward_weights` 显式指定（逗号分隔或 JSON 列表），可选项目前内置：

- `format`：格式检查（非法格式给负分）
- `rule`：严格匹配 ground_truth token ids
- `ndcg`：组内排序奖励（NDCG 风格）

相关参数：

- `--use_prm`：启用 per-token PRM 风格 reward（`rule/ndcg` 会返回 token-level reward 序列）
- `--prm_match_mode position|prefix`：token 对齐策略（逐位置 or 前缀）
- `--noscale` / `--nodemean`：优势归一化细节（是否除 std、是否减均值）

### 4.4 训练（GRPO/PPO 风格）

- `--train_batch_size` / `--eval_batch_size`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--beta`：GRPO loss 里的 KL 系数（越大越“保守”，越贴近 base/ref）
- `--clip` + `--clip_ratio*`：开启 PPO-style ratio clipping（含 dual clip）
- `--use_sft_loss` + `--sft_loss_coef`：在 RL 更新中混入辅助 SFT loss（UFT 风格）

### 4.5 评估与日志

- `--eval_step`：评估频率（steps/ration，取决于 TRL/Trainer 的解释）
- `--eval_on_test` / `--eval_on_valid`：是否在 test/valid split 上评估
- `--test_beam`：测试时 beam size（注意与 `num_generations` 的关系，见下方“坑”）
- `--log_completions` + `--completion_log_interval`：把 rollout 的 prompt/completion/reward 打到 wandb（脚本中常配合 `WANDB_MODE=offline`）

---

## 5. 输出目录与保存内容

RL 训练会在 `--output_dir` 下保存 checkpoint（具体结构与 Trainer/accelerate 配置相关）。

`src/rl/rl.py` 在训练结束后还会额外写：

- `output_dir/final_checkpoint/`：`save_pretrained()` 形式保存最终模型与 tokenizer/config

---

## 6. 常见坑与排查建议

1. **`format_reward` 认为输出格式不合法**
   - `src/rl/reward_fns.py::format_reward` 对 `seqrec/fusionseqrec` 会做正则校验：
     - 不允许换行
     - 目前默认按 `<a_?><b_?><c_?><d_?><|im_end|>` 的形式检查
   - 如果你的 SID 层数不是 4（例如只有 3 层 `<a><b><c>`），需要调整该正则/逻辑，否则大部分 completion 会被打负分。
2. **`num_generations` 与测试 beam 不一致导致 shape 报错**
   - `ConstrainedLogitsProcessor` 会检查 `input_ids.shape[0]` 是否是 `num_beams` 的整数倍。
   - 如果训练时 `num_generations=16`、测试时 `test_beam=20`，要确保 Trainer 内部生成配置与 logits processor 的 beam 参数一致，否则会抛异常。
3. **prefix_index 不对导致前缀约束失效**
   - `src/rl/rl.py` 里会根据 `base_model` 名字粗略设置 `prefix_index`（llava=7，gpt2=4，其他=3）。
   - 如果你换了模型/chat template，可以用 `rl.py` 里的 `debug_prefix_index()` 打印 tokenization 结果，手动调整 `prefix_index` 更稳。
