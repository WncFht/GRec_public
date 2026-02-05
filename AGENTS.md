# AGENTS.md

> 作用：给后续在本仓库工作的 agent 提供“高价值上下文”，减少重复排查。
> 若与系统/开发者/用户实时指令冲突，以更高优先级指令为准。

## 1) 项目速览

- 项目：`GRec`（多模态生成式推荐）
- 主流水线：Embedding → SID → SFT → (可选) RL → Test
- 关键目录：
  - `src/`: 训练、评测核心逻辑
  - `docs/`: 文档入口（`finetune_readme.md`, `test_readme.md`, `rl_readme.md`）
  - `scripts/`: 可运行模板脚本

## 2) 环境约定（非常重要）

### 本机开发环境（常见：无 CUDA）

- 推荐 Python：`3.10`
- 推荐工具：`uv`
- 建议用途：代码开发、参数检查、轻量调试（非正式训练）

### 服务器训练环境（有 CUDA）

- 训练建议在服务器进行
- 推荐安装顺序：
  1. `python3.10 -m venv .venv`
  2. 安装 CUDA 对应 `torch/torchvision/torchaudio`
  3. `pip install -r requirements.txt`

## 3) 依赖文件说明

- `uv.lock` 当前只覆盖了非常少的依赖（接近最小集，不足以完整训练）
- 完整训练依赖以 `requirements.txt` 为准
- 若在 macOS/无 CUDA 本机安装，`requirements.txt` 里的 CUDA/NVIDIA 相关包可能不适配

## 4) 近期关键代码行为（已落地）

1. `torch.compile` 生效顺序
   - 已调整为：**先 compile model，再创建 Trainer**
   - 相关训练入口：`src/finetune/train_ddp*.py`, `src/finetune/train_muon.py`

2. `set_seed` 行为
   - `src/utils.py:set_seed(seed, deterministic=False)`
   - 默认：性能优先（启用 cuDNN）
   - `--deterministic`：更强复现（通常更慢）

3. `model_type` 对齐
   - 支持 `qwen2_instruct`（此前曾有缺失/拼写不一致问题）

4. `load_test_dataset` 多数据集
   - 支持 `--dataset a,b,c` 合并评测
   - 多数据集返回 `TestConcatDataset`，保留 `set_prompt/get_all_items` 接口

5. 验证集缺失处理
   - 若 `save_and_eval_strategy != no` 且无法构建 valid，会明确报错
   - 若 `save_and_eval_strategy == no`，允许无 valid

## 5) 常用快速检查

- 语法检查：
  - `python -m compileall -q src`
- 参数帮助：
  - `python -m src.finetune.train_ddp --help`
  - `python -m src.seqrec.metric --help`

## 6) 提交前建议

- 优先做最小改动，避免改动无关文件
- 不要留下临时文件（如本地专用 `requirements.*.txt`）
- 若改了训练/评测参数语义，同步更新：
  - `docs/finetune_readme.md`
  - `docs/test_readme.md`
  - `README.md`（必要时）

