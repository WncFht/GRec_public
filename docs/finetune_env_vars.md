# SFT 脚本环境变量参数表（含默认值）

本文档覆盖当前 SFT 主流程脚本的环境变量：

- `scripts/finetune/bundle_train_common.sh`
- `scripts/finetune/train_text.sh`
- `scripts/finetune/bundle_metric_common.sh`
- 以及示例包装脚本（如 `train.sh` / `metric.sh` / `train_ep15.sh` / `metric_ep15.sh`）

> 约定：
>
> - 语法 `VAR="${VAR:-default}"` 表示“若未传则使用默认值”。
> - 语法 `: "${VAR:=default}"` 等价于“若未定义则赋默认值”。
> - 包装脚本 `export` 的变量优先级高于公共脚本默认值。

---

## 1) `bundle_train_common.sh` 参数

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `INDEX_TAG` | 无（必填） | 实验索引标签（如 `rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003`），用于定位 index 文件。 |
| `GREC_ROOT` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec` | 项目根目录。 |
| `DATASET` | `Instruments` | 数据集名称。 |
| `DATA_PATH` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data` | 数据根目录。 |
| `ROOT_DIR` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian` | 输出根目录（`ckpt` 所在根）。 |
| `INDEX_EMB_MODEL` | `qwen3-embedding-4B` | 用于匹配 index 文件名的 embedding 模型标识。 |
| `INDEX_DATASETS` | `Instruments` | 用于匹配 index 文件名的数据集标签（支持逗号分隔，会转成 `-`）。 |
| `BASE_MODEL` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/ckpt/base_model/Qwen2.5-3B-Instruct` | 基座模型路径，默认用于推导 `SFT_MODEL_TAG`。 |
| `SFT_MODEL_TAG` | 自动推导 | SFT 模型标签。未传时从 `BASE_MODEL` 的目录名规范化得到（小写、空格/下划线/路径分隔转 `-`）。 |
| `INDEX_FILE` | 自动发现最新 | 若不传，会按 `DATA_PATH/DATASET/${DATASET}.index_emb-${INDEX_EMB_MODEL}_${INDEX_MATCH_TAG}_ds${INDEX_DATASETS_TAG}_rid*.json` 找最新文件。 |
| `TASKS` | `item2index,seqrec,fusionseqrec` | 用于输出目录中的任务 tag。 |
| `RUN_ID` | `$(date +%Y%m%d_%H%M%S)` | 运行唯一标识。 |
| `OUTPUT_DIR` | `${ROOT_DIR}/ckpt/${DATASET}/${SFT_MODEL_TAG}-sft__tasks-${TASKS_TAG}__idx-${INDEX_KEY}__rid-${RUN_ID}` | 训练输出目录。可手工覆盖。 |
| `EVAL_BY_DATASET` | `true` | 是否按数据集拆分 eval。 |
| `EVAL_MAIN_DATASET` | `${DATASET}` | `eval_by_dataset` 下用于主指标的验证集名。 |

---

## 2) `train_text.sh` 参数

### 2.1 运行环境与日志平台

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `WANDB_MODE` | `offline` | W&B 工作模式（`offline` / `online` 等）。 |
| `WANDB_PROJECT` | `grec_sft` | W&B project 名。 |
| `WANDB_ENTITY` | `generate_rec` | W&B entity 名。 |
| `WANDB_NAME` | `sft_text_${DATASET_TAG}__model-${SFT_MODEL_TAG}__tasks-${TASKS_TAG}__idx-${INDEX_KEY}` | W&B run 名。 |
| `PYTHONUNBUFFERED` | `1`（脚本内固定） | Python 无缓冲输出。 |
| `CUDA_LAUNCH_BLOCKING` | `0` | CUDA 同步调试开关。 |
| `PYTHONNOUSERSITE` | `1` | 禁用用户 site-packages。 |

### 2.2 数据与模型

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `GREC_ROOT` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec` | 项目根目录。 |
| `ROOT_DIR` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian` | 数据/权重根目录。 |
| `DATASET` | `Instruments` | 训练数据集（可逗号分隔多数据集）。 |
| `DATA_PATH` | `${ROOT_DIR}/data` | 数据根目录。 |
| `BASE_MODEL` | `${ROOT_DIR}/ckpt/base_model/Qwen2.5-3B-Instruct` | 基座模型目录。 |
| `MODEL_TYPE` | `qwen2_5_instruct` | 传给 `src.finetune.train_ddp` 的模型类型。 |
| `TASKS` | `item2index,seqrec,fusionseqrec` | 训练任务列表。 |
| `INDEX_FILE` | `.index_qwen3-embedding-4B.json` | index 文件后缀（实际拼接为 `data/<dataset>/<dataset><INDEX_FILE>`）。 |
| `SFT_MODEL_TAG` | 自动推导 | 先取已有 `SFT_MODEL_TAG`，否则由 `BASE_MODEL` 目录名规范化得到。 |
| `RUN_ID` | `$(date +%Y%m%d_%H%M%S)` | 运行唯一标识。 |
| `OUTPUT_DIR` | `${ROOT_DIR}/ckpt/${DATASET_TAG}/${SFT_MODEL_TAG}-sft__tasks-${TASKS_TAG}__idx-${INDEX_KEY}__rid-${RUN_ID}` | 训练输出目录。 |
| `CHECK_INDEX_FILES` | `true` | 是否在启动前检查 index 文件存在。 |

### 2.3 分布式与资源

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `GPUS` | `0,1,2,3` | 多卡训练可见 GPU 列表。 |
| `NPROC` | `4` | `torch.distributed.run` 进程数。 |
| `MASTER_PORT` | `33326` | DDP 主端口。 |
| `PYTHON_BIN` | 自动 | 若 `CONDA_PREFIX/bin/python` 可执行则优先；否则 `python`。 |
| `DEBUG_GPU` | `0`（仅 `--debug` 时） | debug 单卡模式使用的 GPU。 |

### 2.4 训练超参数

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `SEED` | `42` | 随机种子。 |
| `PER_DEVICE_BATCH_SIZE` | `8` | 每卡 batch size。 |
| `GRAD_ACC` | `4` | 梯度累积步数。 |
| `NUM_WORKERS` | `16` | DataLoader worker 数。 |
| `LEARNING_RATE` | `5e-5` | 学习率。 |
| `EPOCHS` | `10` | 训练轮数。 |
| `WEIGHT_DECAY` | `0.01` | 权重衰减。 |
| `LR_SCHEDULER_TYPE` | `cosine` | 学习率调度器。 |
| `SAVE_AND_EVAL_STRATEGY` | `epoch` | 保存/评估策略（`epoch` / `steps` / `no`）。 |
| `SAVE_AND_EVAL_STEPS` | `1000` | steps 策略下保存/评估间隔。 |
| `DEEPSPEED_CONFIG` | `${GREC_ROOT}/config/ds_z2_bf16.json` | DeepSpeed 配置文件。 |
| `TRAIN_PROMPT_SAMPLE_NUM` | `1,1,1` | 各任务 prompt 采样数。 |
| `TRAIN_DATA_SAMPLE_NUM` | `0,0,0` | 各任务数据采样数（`0` 常表示全量）。 |
| `RATIO_DATASET` | `1` | 数据集采样比例。 |

### 2.5 训练行为开关

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `USE_LORA` | `false` | 是否启用 LoRA。 |
| `LORA_MODULES_TO_SAVE` | `embed_tokens,lm_head` | LoRA 下额外保存模块。 |
| `FREEZE` | 空 | 冻结参数策略字符串。 |
| `ONLY_TRAIN_RESPONSE` | `true` | 是否仅训练 response 部分。 |
| `USE_GRADIENT_CHECKPOINTING` | `true` | 是否启用梯度检查点。 |
| `REPORT_TO` | `wandb` | Trainer 上报后端（`wandb` / `tensorboard` / `none`）。 |
| `DETERMINISTIC` | `false` | 是否启用 deterministic 训练。 |
| `USE_TORCH_COMPILE` | `false` | 是否启用 `torch.compile`。 |
| `RUN_IN_FOREGROUND` | `false` | `true` 前台运行；`false` 后台 `nohup`。 |
| `EVAL_BY_DATASET` | `true` | 是否按数据集分开 eval。 |
| `EVAL_MAIN_DATASET` | `Instruments` | 主验证集名称。 |

### 2.6 命令行开关（非环境变量）

| 参数 | 含义 |
|---|---|
| `--debug` | 进入前台单卡 debug 流程，不走 DDP 多卡后台。 |

---

## 3) `bundle_metric_common.sh` 参数

### 3.1 评测核心参数

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `INDEX_TAG` | 无（必填） | 与训练使用同源 index 的标签。 |
| `GREC_ROOT` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GRec` | 项目根目录。 |
| `TASK` | `seqrec` | 测试任务名（`--test_task`）。 |
| `DATASET` | `Instruments` | 测试数据集。 |
| `RATIO` | `1` | 评测数据比例（`--ratio_dataset`）。 |
| `MODEL_TYPE` | `qwen2_5_instruct` | 模型类型（传给 metric 程序）。 |
| `DATA_PATH` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data` | 数据根目录。 |
| `ROOT_DIR` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian` | ckpt 根目录。 |
| `INDEX_EMB_MODEL` | `qwen3-embedding-4B` | index 文件匹配用 embedding 标识。 |
| `INDEX_DATASETS` | `Instruments` | index 文件匹配用数据集标签。 |
| `BASE_MODEL` | `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/ckpt/base_model/Qwen2.5-3B-Instruct` | 用于推导 `SFT_MODEL_TAG`。 |
| `SFT_MODEL_TAG` | 自动推导 | 评测查找 SFT ckpt 目录的模型标签。 |
| `INDEX_FILE` | 自动发现最新 | 若不传，按 index 模式自动找最新。 |
| `CKPT_PATH` | 自动发现最新 | 若不传，按 `${ROOT_DIR}/ckpt/${DATASET}/${SFT_MODEL_TAG}-sft__tasks-${TASKS_TAG}__idx-${INDEX_KEY}__rid-*` 找最近 run，再取最新 `checkpoint-*`。 |
| `SFT_DIR` | 空 | 手工指定某次训练目录；优先于自动查找。 |
| `TASKS` | `item2index,seqrec,fusionseqrec` | 仅在自动解析 `CKPT_PATH` 时用于匹配 SFT 目录。 |

### 3.2 推理/评测资源与结果输出

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `NUM_GPUS` | `4` | `torchrun --nproc_per_node`。 |
| `MASTER_PORT` | `33320` | DDP 端口。 |
| `BATCH_SIZE` | `16` | 测试 batch size。 |
| `NUM_BEAMS` | `50` | beam search 宽度。 |
| `MAX_NEW_TOKENS` | `4` | 最大生成 token 数。 |
| `EVAL_SPLIT` | `test` | 评测 split。 |
| `RESULTS_BASE_DIR` | `./results/${EVAL_SPLIT}` | 结果根目录。 |
| `RESULTS_FILE` | `${RUN_DIR}/results.json` | 指标输出文件。 |
| `ROLLOUT_FILE` | `${RUN_DIR}/rollout.json` | rollout 输出文件。 |
| `LOG_FILE` | `${RUN_DIR}/log.txt` | 后台日志文件。 |
| `TEST_PROMPT_IDS` | `0` | 测试 prompt id 列表。 |
| `METRICS` | `hit@1,hit@3,hit@5,hit@10,hit@20,hit@50,ndcg@1,ndcg@3,ndcg@5,ndcg@10,ndcg@20,ndcg@50` | 评测指标列表。 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 评测可见 GPU。 |

### 3.3 命令行开关（非环境变量）

| 参数 | 含义 |
|---|---|
| `--debug` | 前台执行 metric（不 `nohup`）。 |
| `--force-rollout` | 透传 `--force_rollout`。 |
| `--skip-rollout` | 透传 `--skip_rollout`。 |

---

## 4) 包装脚本默认导出（示例）

### 4.1 常规 bundle 包装脚本

示例：`scripts/finetune/Instruments_.../train.sh`、`metric.sh`

- `train.sh`：通常只固定 `INDEX_TAG`，其余参数由公共脚本默认值或外部 env 控制。
- `metric.sh`：通常只固定 `INDEX_TAG`，其余参数由公共脚本默认值或外部 env 控制。

### 4.2 `ep15` 示例脚本

- `train_ep15.sh` 默认额外导出：
  - `EPOCHS=15`
  - `RUN_ID=ep15_<timestamp>`
- `metric_ep15.sh` 默认额外导出：
  - `TASKS=item2index,seqrec,fusionseqrec`（用于匹配 SFT 目录）

---

## 5) 推荐覆盖方式

训练（7B instruct 例子）：

```bash
BASE_MODEL=/mnt/.../ckpt/base_model/Qwen2.5-7B-Instruct \
MODEL_TYPE=qwen2_5_instruct \
WANDB_MODE=online \
bash scripts/finetune/Instruments_Instruments-qwen3-embedding-4B-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003/train_ep15.sh
```

评测（与训练同模型 tag）：

```bash
BASE_MODEL=/mnt/.../ckpt/base_model/Qwen2.5-7B-Instruct \
TASK=seqrec \
bash scripts/finetune/Instruments_Instruments-qwen3-embedding-4B-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003/metric_ep15.sh
```

如需固定某次模型，建议显式给 `CKPT_PATH` 或 `SFT_DIR`。
