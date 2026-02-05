# 集群并行友好的运行脚本生成与命名规范

> 目标：避免“所有实验都从同一个脚本入口启动”带来的并发冲突问题，
> 通过**每次实验自动生成独立运行脚本**，统一管理 `output_dir / ckpt / wandb / log` 命名。

---

## 1. 适用场景与设计目标

适用于以下场景：

- 在集群上并行提交多个 SFT / RL 任务
- 需要快速定位“某个 checkpoint 对应哪次实验参数”
- 希望 wandb 名称与本地日志、脚本文件一一对应

设计目标：

1. **每个任务一个脚本快照**：提交后可复现实验，不依赖“共享入口脚本”后续改动。
2. **命名可读 + 可排序 + 低冲突**：看名字就知道关键配置，并尽量避免重名。
3. **分层命名**：
   - `RUN_KEY`：稳定实验签名（便于分组对比）
   - `RUN_ID`：单次运行实例（唯一，便于追踪）
4. **统一落盘**：脚本、日志、ckpt、元信息都有固定目录结构。

---

## 2. 核心约定：不要直接“共享脚本即实验本体”

推荐流程：

1. 维护模板脚本（如 `scripts/finetune/train_text.sh`、`scripts/finetune/train_vl.sh`、`scripts/rl/rl.sh`）
2. 每次运行前，用配置生成一个**独立的运行脚本**（generated run script）
3. 提交的永远是这个 generated 脚本，而不是模板本身

这样即使模板后续被修改，历史实验仍可完全复现。

---

## 3. 命名规则（重点）

## 3.1 字段定义

- `stage`：`sft_text` / `sft_vl` / `rl`
- `dataset_slug`：从 `--dataset` 规范化得到
- `task_slug`：从 `--tasks` 规范化得到
- `model_slug`：通常来自 `BASE_MODEL` 的 basename
- `index_slug`：来自 `INDEX_FILE`
- `seed`：随机种子
- `time_tag`：`YYYYMMDD-HHMMSS`
- `git_tag`：`g<short_sha>`（如 `g1a2b3c4`）
- `job_tag`：`j<SLURM_JOB_ID>` 或 `jlocal`

## 3.2 规范化规则（slugify）

对每个字段做统一清洗：

- 全部转小写
- 空格、逗号、斜杠转 `-`
- 连续多个 `-` 压缩为一个
- 去掉首尾 `-`
- 建议限制长度（例如 24 或 32），超长截断

示例：

- `Instruments` → `instruments`
- `Arts,Games,Instruments` → `arts-games-instruments`
- `Qwen2.5-7B-Instruct` → `qwen2-5-7b-instruct`

## 3.3 RUN_KEY（稳定实验签名）

建议格式：

```text
<stage>__<dataset_slug>__<task_slug>__<model_slug>__idx-<index_slug>__s<seed>
```

示例：

```text
sft_text__instruments__item2index-seqrec__qwen2-5-3b-instruct__idx-index-qwen3-embedding-4b__s42
```

用途：

- `WANDB_GROUP`
- 分组目录（可选）
- 对比实验的“同一族”标识

## 3.4 RUN_ID（单次运行唯一 ID）

建议格式：

```text
<RUN_KEY>__<time_tag>__<git_tag>__<job_tag>
```

示例：

```text
sft_text__instruments__item2index-seqrec__qwen2-5-3b-instruct__idx-index-qwen3-embedding-4b__s42__20260205-231530__g1a2b3c4__j428901
```

用途：

- `WANDB_NAME`
- `OUTPUT_DIR` 末级目录名
- 日志文件名
- 生成脚本文件名

---

## 4. 目录规范

建议新增统一目录（与现有结构兼容）：

```text
runs/
  generated/<stage>/<yyyymmdd>/<RUN_ID>.sh      # 自动生成的可提交脚本
  specs/<stage>/<yyyymmdd>/<RUN_ID>.env         # 本次运行的参数快照（可选）
  meta/<stage>/<yyyymmdd>/<RUN_ID>.txt          # git sha / 主机 / 提交命令（可选）

log/<stage>/<dataset_slug>/<yyyymmdd>/<RUN_ID>.log
ckpt/<dataset_slug>/<stage>/<RUN_ID>/
```

说明：

- `generated` 与 `ckpt/log` 一对一对应，排查非常快。
- 即使同一 `RUN_KEY` 并行提交，也会因为 `time_tag + job_tag` 区分开。

---

## 5. wandb 命名规则

建议统一如下：

- `WANDB_PROJECT`
  - SFT：`GRec-sft`
  - RL：`GRec-rl`
- `WANDB_GROUP`：`RUN_KEY`
- `WANDB_NAME`：`RUN_ID`
- `WANDB_TAGS`（可选）：`stage,dataset,task,model,seed,index`

这样可同时满足：

- 同组对比（看 `group`）
- 单次运行追踪（看 `name`）
- 跨项目隔离（看 `project`）

---

## 6. OUTPUT_DIR / CKPT 命名规则

统一建议：

```text
OUTPUT_DIR=ckpt/<dataset_slug>/<stage>/<RUN_ID>
```

例如：

```text
ckpt/instruments/sft_text/sft_text__instruments__item2index-seqrec__qwen2-5-3b-instruct__idx-index-qwen3-embedding-4b__s42__20260205-231530__g1a2b3c4__j428901
```

附加建议：

- 不要把“可变参数”只写在文件夹注释里，必须进入 `RUN_KEY` 或 `RUN_ID`。
- 每次运行把完整命令写入 `OUTPUT_DIR/cmd.sh`（便于复现）。
- 训练结束后写 `OUTPUT_DIR/run_meta.txt`（记录 git sha、hostname、job id、开始结束时间）。

---

## 7. 生成脚本建议流程（SFT / RL 通用）

## 7.1 输入

最少输入参数：

- `stage`、`dataset`、`tasks`、`base_model`、`index_file`、`seed`
- 分布式参数：`gpus/nproc/port`（SFT）或 `num_processes/main_process_port`（RL）
- 训练超参（batch/lr/epochs 等）

## 7.2 生成逻辑

1. 从输入生成 `dataset_slug/task_slug/model_slug/index_slug`
2. 计算 `RUN_KEY`
3. 采集 `time_tag/git_tag/job_tag` 生成 `RUN_ID`
4. 衍生路径与 wandb 名称：
   - `OUTPUT_DIR`
   - `LOG_FILE`
   - `WANDB_NAME/GROUP`
5. 渲染为一个完整 `.sh`（包含所有实际参数，而不是只依赖外部环境）

## 7.3 输出

- `runs/generated/.../<RUN_ID>.sh`
- `runs/specs/.../<RUN_ID>.env`（可选）
- `runs/meta/.../<RUN_ID>.txt`（可选）

---

## 8. 生成脚本模板示例

## 8.1 SFT（示意）

```bash
#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT="GRec-sft"
export WANDB_NAME="${RUN_ID}"
export WANDB_GROUP="${RUN_KEY}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export CUDA_VISIBLE_DEVICES="${GPUS}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  -m src.finetune.train_ddp \
  --seed "${SEED}" \
  --model_type "${MODEL_TYPE}" \
  --base_model "${BASE_MODEL}" \
  --dataset "${DATASET}" \
  --tasks "${TASKS}" \
  --index_file "${INDEX_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  ...
```

## 8.2 RL（示意）

```bash
#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT="GRec-rl"
export WANDB_NAME="${RUN_ID}"
export WANDB_GROUP="${RUN_KEY}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export CUDA_VISIBLE_DEVICES="${GPUS}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --module src.rl.rl \
  --model_type "${MODEL_TYPE}" \
  --base_model "${BASE_MODEL}" \
  --dataset "${DATASET}" \
  --tasks "${TASK}" \
  --index_file "${INDEX_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  ...
```

---

## 9. 集群调度集成建议（Slurm 示例）

推荐把 generated 脚本作为真正提交对象：

```bash
sbatch --job-name "${RUN_ID}" runs/generated/sft_text/20260205/${RUN_ID}.sh
```

约定：

- `job_name` 使用 `RUN_ID` 或其截断版本
- 如果调度器会注入作业 ID，可在脚本里记录：
  - `SLURM_JOB_ID`
  - `SLURM_NODELIST`
  - `SLURM_GPUS`

---

## 10. 反模式（建议避免）

1. 所有实验都复用同一个 `OUTPUT_DIR`。
2. wandb 全部写同一个 `WANDB_NAME`（曲线被覆盖、难追溯）。
3. 同一模板脚本反复手改后直接提交（历史不可复现）。
4. `RUN_ID` 不包含时间或作业标识，导致并行冲突。

---

## 11. 最小落地清单

建议先做到这 5 条（收益最高）：

1. 引入 `RUN_KEY` + `RUN_ID` 双层命名
2. `WANDB_GROUP=RUN_KEY`，`WANDB_NAME=RUN_ID`
3. `OUTPUT_DIR=ckpt/<dataset>/<stage>/<RUN_ID>`
4. 每次提交前生成独立 `.sh` 到 `runs/generated/`
5. 把实际运行命令落盘到 `OUTPUT_DIR/cmd.sh`

---

## 12. 与当前仓库脚本的关系

- `scripts/finetune/train_text.sh` / `train_vl.sh` / `scripts/rl/rl.sh` 仍可作为“参数模板来源”。
- 在集群场景中，建议再加一层“脚本生成器”来产出最终提交脚本（本规范即该生成器的命名与目录约束）。
- 这样既保留模板维护便利性，也满足并发实验的可追踪与可复现。


---

## 13. 仓库内已落地生成器

当前仓库已提供脚本生成器：

- `scripts/tools/gen_run.sh`

你可以直接用它按本规范生成运行脚本（含 `RUN_KEY / RUN_ID / output_dir / log / wandb` 命名）。

### 13.1 SFT 示例

```bash
bash scripts/tools/gen_run.sh \
  --stage sft_text \
  --dataset Instruments \
  --tasks item2index,seqrec \
  --base-model ckpt/base_model/Qwen2.5-3B-Instruct \
  --model-type qwen2_5_instruct \
  --index-file .index_qwen3-embedding-4B.json \
  --set PER_DEVICE_BATCH_SIZE=8 \
  --set GRAD_ACC=4
```

### 13.2 RL 示例

```bash
bash scripts/tools/gen_run.sh \
  --stage rl \
  --dataset Instruments \
  --tasks seqrec \
  --base-model ckpt/Instruments/qwen2.5-sft/checkpoint-1234 \
  --model-type qwen2_5_instruct \
  --index-file .index_qwen3-embedding-4B.json \
  --set REWARD_TYPE=ranking \
  --set NUM_GENERATIONS=16
```

### 13.3 产物位置

- 运行脚本：`runs/generated/<stage>/<yyyymmdd>/<RUN_ID>.sh`
- 参数快照：`runs/specs/<stage>/<yyyymmdd>/<RUN_ID>.env`
- 元信息：`runs/meta/<stage>/<yyyymmdd>/<RUN_ID>.txt`

