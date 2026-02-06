# `scripts/rl` 使用说明

本目录同样分为两类：

1. **统一模板入口（推荐）**
   - `rl.sh`：RL 训练统一入口（`accelerate launch -m src.rl.rl`）
2. **历史实验脚本（保留）**
   - `train_seqrec_*.sh`、`qwen2.5_instruct_*.sh`、`hope/*.sh`
   - 含特定实验配置/路径，供复现实验记录用

---

## 快速用法

默认后台启动：

```bash
bash scripts/rl/rl.sh
```

调试模式（前台单卡）：

```bash
bash scripts/rl/rl.sh --debug
```

---

## 配置方式（推荐环境变量）

```bash
DATASET=Instruments \
DATA_PATH=./data \
BASE_MODEL=ckpt/Instruments/qwen2.5-sft/checkpoint-1234 \
MODEL_TYPE=qwen2_5_instruct \
OUTPUT_DIR=ckpt/Instruments/qwen2.5-rl__idx-index_emb-qwen3-embedding-4B_rq4_cb512-512-512-512_dsInstruments_ridJan-28-2026-05-54-58 \
TASK=seqrec \
INDEX_FILE=.index_emb-qwen3-embedding-4B_rq4_cb512-512-512-512_dsInstruments_ridJan-28-2026-05-54-58.json \
REWARD_TYPE=ranking \
TRAIN_BATCH_SIZE=64 \
EVAL_BATCH_SIZE=128 \
NUM_TRAIN_EPOCHS=2 \
GRAD_ACC=2 \
NUM_GENERATIONS=16 \
LEARNING_RATE=1e-5 \
BETA=1e-3 \
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=29503 \
bash scripts/rl/rl.sh
```

常用变量：

- 运行资源：`GPUS`, `NUM_PROCESSES`, `MAIN_PROCESS_PORT`, `ACCELERATE_CONFIG`
- 数据/模型：`DATASET`, `DATA_PATH`, `BASE_MODEL`, `MODEL_TYPE`, `OUTPUT_DIR`, `TASK`, `INDEX_FILE`
- 默认 `OUTPUT_DIR` 与日志名会自动带上 `INDEX_FILE` key（避免混 run）
- 脚本会先检查 `data/<dataset>/<dataset><INDEX_FILE>` 是否存在
- 训练超参：`TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `NUM_TRAIN_EPOCHS`, `GRAD_ACC`, `LEARNING_RATE`, `BETA`
- rollout / 解码：`NUM_GENERATIONS`, `USE_BEAM_SEARCH`, `MAX_COMPLETION_LENGTH`, `TEMPERATURE`
- 训练行为：`TEST_DURING_TRAINING`, `EVAL_ON_TEST`, `LOG_COMPLETIONS`, `COMPLETION_LOG_INTERVAL`, `DETERMINISTIC`

---

## 建议

- 新 RL 实验优先从 `rl.sh` 起步，先用环境变量改参数。
- 某组固定配置可另建极简 wrapper（只负责导出环境变量并调用 `rl.sh`）。
- `scripts/rl/hope/*.sh` 暂保留，后续确认不再使用后再清理。

