# SeqRec 结果汇总（GRec + GenRec）

- 生成日期: 2026-02-18
- GRec 数据源: `/Users/fanghaotian/Desktop/src/GRec_public/results/test/seqrec-constrained`（11 runs）
- GenRec 数据源: `/Users/fanghaotian/Desktop/src/GenRec/results`（8 runs）
- 汇总表: `GRec_public/docs/seqrec_aggregated_metrics_2026-02-18.tsv`

## 自动生成脚本

- 脚本: `GRec_public/scripts/tools/generate_seqrec_summary.py`
- 命令:
```bash
python3 GRec_public/scripts/tools/generate_seqrec_summary.py --workspace-root /Users/fanghaotian/Desktop/src
```

## 可比性说明（重要）

- `GenRec` 的前两条结果是 `Industrial_and_Scientific` 数据集，不应直接与 `Instruments` 结果混比。
- `Industrial_and_Scientific-qwen2.5-3b-instruct-grpo` 可作为 `Industrial_and_Scientific-sft-dsz0-4gpu-eq8` 的 SFT 后续 RL（GRPO）结果对比。
- `GenRec/Instruments-grec-sft-*` 与 `GenRec/Instruments-mimionerec-sft-*` 使用相同模型与训练超参框架，主要变量是数据切分策略。
- `GenRec/scripts/prepare_category_from_inter_json.py` 中，`grec` 采用 per-user leave-2-out；`mimionerec` 先构造全量 next-item 样本再做全局 8:1:1 切分。
- `GenRec/hope/*-evaluate.sh` 显示两组评测分别读取各自数据变体的 `test.json`，因此绝对值差异应视为“切分+训练”联合效果。
- `cb64` 组内三条 run 不是同配方：第一条为 `ep15`；后两条按实验说明是 `ep10`，且任务配置不同（`seqrec+item2index+fusionseqrec` vs `seqrec+item2index`），所以 checkpoint step 不能直接横向对齐。
- `GenRec` 任务构成来自 `GenRec/preprocess_data_sft_rl.py:474` 与 `GenRec/preprocess_data_sft_rl.py:540`：SFT 使用 `task1+task2+task3`，RL 使用 `task1+task4+task5`。

## GenRec 任务定义（task1~task5）

- 来源: `GenRec/preprocess_data_sft_rl.py:479`, `GenRec/preprocess_data_sft_rl.py:497`, `GenRec/preprocess_data_sft_rl.py:502`, `GenRec/preprocess_data_sft_rl.py:508`, `GenRec/preprocess_data_sft_rl.py:517`

| Task | Definition | Used In | RL Ability |
| --- | --- | --- | --- |
| task1_sid_sft | 顺序推荐主任务：输入历史交互序列（semantic IDs），预测下一个 semantic ID。 | SFT + RL | seq_rec |
| task2_sid_item_feat | item 属性问答：sid->title 与 title->sid 双向 QA。 | SFT only | - |
| task3_fusion_seq | FusionSeqRec：输入历史 semantic IDs，预测下一物品 title。 | SFT only | - |
| task4_hisTitle2sid | Title2Sid 序列任务：输入历史 item title 序列，预测下一项 semantic ID。 | RL only | seq_title2sid |
| task5_title_desc2sid | Title/Description2Sid：给 title 或 description，预测 semantic ID。 | RL only | title_desc2sid |

## GRec 总榜（按 NDCG@10）

| Rank | Group | Checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `qwen2.5_instruct_seqrec_ranking` | `checkpoint-5306` | 0.0983 | 0.1250 | 0.1095 | 0.1754 |
| 2 | `Qwen2.5-3B-Instruct-sft-index_qwen3-embedding-4B-multi` | `checkpoint-11645` | 0.0972 | 0.1234 | 0.1164 | 0.2127 |
| 3 | `Qwen2.5-7B-Instruct-sft-index_qwen3-embedding-4B-5e-5` | `checkpoint-12294` | 0.0955 | 0.1209 | 0.1118 | 0.1947 |
| 4 | `Instruments-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003` | `checkpoint-20490` | 0.0950 | 0.1248 | 0.1156 | 0.2195 |
| 5 | `Instruments-rq4_cb512-512-512-512_sk0.0-0.0-0.0-0.003` | `checkpoint-13660` | 0.0940 | 0.1258 | 0.1140 | 0.2179 |
| 6 | `Instruments-rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003` | `checkpoint-13660` | 0.0919 | 0.1266 | 0.1125 | 0.2217 |
| 7 | `Instruments-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003` | `checkpoint-13660` | 0.0911 | 0.1180 | 0.1117 | 0.2131 |
| 8 | `Instruments-rq4_cb128-128-128-128_sk0.0-0.0-0.0-0.003` | `checkpoint-13660` | 0.0833 | 0.1182 | 0.1033 | 0.2106 |
| 9 | `Instruments-rq4_cb32-32-32-32_sk0.0-0.0-0.0-0.003` | `checkpoint-13660` | 0.0820 | 0.1101 | 0.1021 | 0.2028 |
| 10 | `Instruments-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003` | `checkpoint-4921` | 0.0667 | 0.0941 | 0.0852 | 0.1793 |
| 11 | `Qwen2.5-3B-Instruct-multi-sft-index_qwen3-embedding-4B` | `checkpoint-43820` | 0.0602 | 0.0847 | 0.0754 | 0.1554 |

## RQ4 Codebook Sweep（Instruments, instruct+tasks-item2index-seqrec-fusionseqrec）

| Codebook | NDCG@10 | HR@10 | NDCG@50 | HR@50 | Delta vs cb32 |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.0820 | 0.1101 | 0.1021 | 0.2028 | +0.0000 |
| 64 | 0.0950 | 0.1248 | 0.1156 | 0.2195 | +0.0130 |
| 128 | 0.0833 | 0.1182 | 0.1033 | 0.2106 | +0.0012 |
| 256 | 0.0919 | 0.1266 | 0.1125 | 0.2217 | +0.0098 |
| 512 | 0.0940 | 0.1258 | 0.1140 | 0.2179 | +0.0120 |

## cb64 组内对比（带实验注释）

| Model Variant | Epoch Hint | Task Hint | Step | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b-instruct-sft__tasks-item2index-seqrec-fusionseqrec__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11__rid-ep15_20260212_012019` | ep15 (run id) | item2index+seqrec+fusionseqrec | 20490 | 0.0950 | 0.1248 | 0.1156 | 0.2195 |
| `qwen2.5-3b-sft__tasks-item2index-seqrec-fusionseqrec__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11__rid-20260211_055856` | ep10 (experiment note) | item2index+seqrec+fusionseqrec | 13660 | 0.0911 | 0.1180 | 0.1117 | 0.2131 |
| `qwen2.5-3b-sft__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11` | ep10 (experiment note) | seqrec+item2index (experiment note) | 4921 | 0.0667 | 0.0941 | 0.0852 | 0.1793 |

## GenRec: Industrial_and_Scientific（仅组内比较）

| Run | Checkpoint | Stage | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | --- | --- | --- | --- | --- |
| `Industrial_and_Scientific-qwen2.5-3b-instruct-grpo` | `checkpoint-495` | sft->rl(grpo) | 0.1008 | 0.1405 | 0.1191 | 0.2255 |
| `Industrial_and_Scientific-sft-dsz0-4gpu-eq8` | `checkpoint-320` | sft | 0.0838 | 0.1193 | 0.1025 | 0.2052 |

- `GRPO - SFT` 增益：`NDCG@10 +0.0170`，`HR@10 +0.0212`。

## GenRec: Instruments（单独列出，不与 Industrial 混比）

| Run | Checkpoint | Config Note | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | --- | --- | --- | --- | --- |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-500` | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.1220 | 0.1626 | 0.1422 | 0.2548 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-300` | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.1158 | 0.1533 | 0.1343 | 0.2378 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-250` | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.1051 | 0.1463 | 0.1243 | 0.2345 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-200` | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.0913 | 0.1331 | 0.1084 | 0.2131 |
| `Instruments-grec-sft-qwen4B-4-256-dsz0` | `checkpoint-495` | cb4-256, qwen3-4B emb, split=grec(leave-2-out) | 0.0823 | 0.1094 | 0.0985 | 0.1844 |
| `Instruments-grec-sft-qwen4B-4-256-dsz0` | `checkpoint-630` | cb4-256, qwen3-4B emb, split=grec(leave-2-out) | 0.0706 | 0.0964 | 0.0850 | 0.1635 |

## Instruments: grec vs mimionerec 深入对比（GenRec）

### 配置与数据构造差异

| Variant | Split Strategy | Preprocess | Train YAML | Eval Script |
| --- | --- | --- | --- | --- |
| `grec` | per-user leave-2-out | `GenRec/scripts/run_instruments_preprocess_grec.sh` | `GenRec/examples/train_full/Instruments/instruments_rec_full_sft_3b_dsz0_qwen4b_4_256_grec.yaml` | `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-evaluate.sh` |
| `mimionerec` | global next-item pool + ratio split (8:1:1) | `GenRec/scripts/run_instruments_preprocess_mimionerec.sh` | `GenRec/examples/train_full/Instruments/instruments_rec_full_sft_3b_dsz0_qwen4b_4_256_mimionerec.yaml` | `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-mimionerec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-mimionerec-evaluate.sh` |

- 两份 YAML 的 backbone、batch size、LR、epoch、deepspeed 基本一致；核心实验变量是数据切分。

### 最优 checkpoint 对比

| Variant | Best Checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | --- | --- | --- | --- |
| `grec` | `checkpoint-495` | 0.0823 | 0.1094 | 0.0985 | 0.1844 |
| `mimionerec` | `checkpoint-500` | 0.1220 | 0.1626 | 0.1422 | 0.2548 |

- `mimionerec(best) - grec(best)`: `NDCG@10 +0.0397` (`+48.2382%`), `HR@10 +0.0532` (`+48.6289%`), `NDCG@50 +0.0437`，`HR@50 +0.0704`。

### Checkpoint 轨迹

| Variant | Checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 | ΔNDCG@10 vs prev | ΔHR@10 vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `grec` | `checkpoint-495` | 0.0823 | 0.1094 | 0.0985 | 0.1844 | - | - |
| `grec` | `checkpoint-630` | 0.0706 | 0.0964 | 0.0850 | 0.1635 | -0.0117 | -0.0130 |
| `mimionerec` | `checkpoint-200` | 0.0913 | 0.1331 | 0.1084 | 0.2131 | - | - |
| `mimionerec` | `checkpoint-250` | 0.1051 | 0.1463 | 0.1243 | 0.2345 | +0.0138 | +0.0132 |
| `mimionerec` | `checkpoint-300` | 0.1158 | 0.1533 | 0.1343 | 0.2378 | +0.0107 | +0.0070 |
| `mimionerec` | `checkpoint-500` | 0.1220 | 0.1626 | 0.1422 | 0.2548 | +0.0062 | +0.0093 |

- `grec` 最近一次从 `checkpoint-495` 到 `checkpoint-630`：`NDCG@10 -0.0117`，`HR@10 -0.0130`。
- `mimionerec` 当前已记录 checkpoints 上 `NDCG@10` 与 `HR@10` 呈单调上升。

## Instruments 交叉框架对比（仅作参考）

- GRec 最优（Instruments）: `GRec_public/results/test/seqrec-constrained/Instruments-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003/qwen2.5-3b-instruct-sft__tasks-item2index-seqrec-fusionseqrec__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11__rid-ep15_20260212_012019/checkpoint-20490/results.json`, NDCG@10=0.0950
- GenRec 最优（Instruments）: `GenRec/results/Instruments-mimionerec-sft-qwen4B-4-256-dsz0/checkpoint-500/metrics.json`, NDCG@10=0.1220
- 差值 (GRec - GenRec): `-0.0270`
