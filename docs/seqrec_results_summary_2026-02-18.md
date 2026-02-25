# SeqRec 结果汇总（GRec + GenRec）

- 生成日期: 2026-02-24
- GRec 数据源: `/Users/fanghaotian/Desktop/src/GRec_public/results/test/seqrec-constrained`（11 runs）
- GenRec 数据源: `/Users/fanghaotian/Desktop/src/GenRec/results`（30 runs）
- 汇总表: `GRec_public/docs/seqrec_aggregated_metrics_2026-02-18.tsv`

## 目录

1. 总体结论（TL;DR）
1. 可比性说明（重要）
1. GenRec 任务定义（task1~task5）
1. GRec 总榜（按 NDCG@10）
1. RQ4 Codebook Sweep（Instruments, instruct+tasks-item2index-seqrec-fusionseqrec）
1. cb64 组内对比（带实验注释）
1. GenRec: Industrial_and_Scientific（仅组内比较）
1. GenRec: Instruments（单独列出，不与 Industrial 混比）
1. GenRec: Instruments-grec SFT Codebook Sweep
1. GenRec: Instruments-grec RL 轨迹（GRPO，含 checkpoint-0 基线）
1. 训练框架 × Codebook 一致性分析（Instruments）
1. Instruments: grec vs mimionerec 深入对比（GenRec, cb256 SFT）
1. Instruments 交叉框架对比（仅作参考）

## 自动生成脚本

- 脚本: `GRec_public/scripts/tools/generate_seqrec_summary.py`
- 命令:
```bash
python3 GRec_public/scripts/tools/generate_seqrec_summary.py --workspace-root /Users/fanghaotian/Desktop/src
```

## 总体结论（TL;DR）

### 本节结论

- GRec 当前全表最优为 `qwen2.5_instruct_seqrec_ranking/checkpoint-5306` (`NDCG@10=0.0983`, `HR@10=0.1250`)。
- GenRec 当前 Instruments 最优为 `Instruments-mimionerec-sft-qwen4B-4-256-dsz0/checkpoint-500` (`NDCG@10=0.1220`, `HR@10=0.1626`)。
- Industrial_and_Scientific 上，`best(GRPO)-best(SFT)` 为 `NDCG@10 +0.0079`，`HR@10 +0.0018`。

## 可比性说明（重要）

- `GenRec` 同时包含 `Industrial_and_Scientific` 与 `Instruments` 两个数据集结果，应先做组内比较，再做跨组参考。
- `Industrial_and_Scientific-qwen2.5-3b-instruct-grpo` 可作为 `Industrial_and_Scientific-sft-dsz0-4gpu-eq8` 的 SFT 后续 RL（GRPO）结果对比。
- `GenRec/Instruments-grec-sft-*` 与 `GenRec/Instruments-mimionerec-sft-*` 使用相同模型与训练超参框架，主要变量是数据切分策略。
- `GenRec/scripts/prepare_category_from_inter_json.py` 中，`grec` 采用 per-user leave-2-out；`mimionerec` 先构造全量 next-item 样本再做全局 8:1:1 切分。
- `GenRec/hope/*-evaluate.sh` 显示两组评测分别读取各自数据变体的 `test.json`，因此绝对值差异应视为“切分+训练”联合效果。
- `cb64` 组内三条 run 不是同配方：第一条为 `ep15`；后两条按实验说明是 `ep10`，且任务配置不同（`seqrec+item2index+fusionseqrec` vs `seqrec+item2index`），所以 checkpoint step 不能直接横向对齐。
- `GenRec` 任务构成来自 `GenRec/preprocess_data_sft_rl.py:474` 与 `GenRec/preprocess_data_sft_rl.py:540`：SFT 使用 `task1+task2+task3`，RL 使用 `task1+task4+task5`。

### 本节结论

- 跨框架比较时需要先按数据集分层，再按任务定义与 split 方式对齐，避免把不可比差异当作模型差异。
- Instruments 上的 `grec`/`mimionerec` 与 RL/SFT 结论都应视为“切分策略 + 训练目标”的联合结果。

## GenRec 任务定义（task1~task5）

- 来源: `GenRec/preprocess_data_sft_rl.py:479`, `GenRec/preprocess_data_sft_rl.py:497`, `GenRec/preprocess_data_sft_rl.py:502`, `GenRec/preprocess_data_sft_rl.py:508`, `GenRec/preprocess_data_sft_rl.py:517`

| Task | Definition | Used In | RL Ability |
| --- | --- | --- | --- |
| task1_sid_sft | 顺序推荐主任务：输入历史交互序列（semantic IDs），预测下一个 semantic ID。 | SFT + RL | seq_rec |
| task2_sid_item_feat | item 属性问答：sid->title 与 title->sid 双向 QA。 | SFT only | - |
| task3_fusion_seq | FusionSeqRec：输入历史 semantic IDs，预测下一物品 title。 | SFT only | - |
| task4_hisTitle2sid | Title2Sid 序列任务：输入历史 item title 序列，预测下一项 semantic ID。 | RL only | seq_title2sid |
| task5_title_desc2sid | Title/Description2Sid：给 title 或 description，预测 semantic ID。 | RL only | title_desc2sid |

### 本节结论

- GenRec 的 SFT 与 RL 并非同一任务集合，RL 的收益应解读为 `task1+task4+task5` 下的能力重分配，而非纯粹延续 SFT 指标。

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

### 本节结论

- 全量 GRec 结果中，`qwen2.5_instruct_seqrec_ranking/checkpoint-5306` 当前为 NDCG@10 最优。
- 在 Instruments 子集里，`Instruments-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003/checkpoint-20490` 是当前最优参考点。

## RQ4 Codebook Sweep（Instruments, instruct+tasks-item2index-seqrec-fusionseqrec）

| Codebook | NDCG@10 | HR@10 | NDCG@50 | HR@50 | Delta vs cb32 |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.0820 | 0.1101 | 0.1021 | 0.2028 | +0.0000 |
| 64 | 0.0950 | 0.1248 | 0.1156 | 0.2195 | +0.0130 |
| 128 | 0.0833 | 0.1182 | 0.1033 | 0.2106 | +0.0012 |
| 256 | 0.0919 | 0.1266 | 0.1125 | 0.2217 | +0.0098 |
| 512 | 0.0940 | 0.1258 | 0.1140 | 0.2179 | +0.0120 |

### 本节结论

- 在当前 GRec RQ4 sweep 中，`Instruments-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003/checkpoint-20490` 最优，`Instruments-rq4_cb32-32-32-32_sk0.0-0.0-0.0-0.003/checkpoint-13660` 最弱。
- 最优与最弱的 `NDCG@10` 差值为 `+0.0130`。

## cb64 组内对比（带实验注释）

| Model Variant | Epoch Hint | Task Hint | Step | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b-instruct-sft__tasks-item2index-seqrec-fusionseqrec__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11__rid-ep15_20260212_012019` | ep15 (run id) | item2index+seqrec+fusionseqrec | 20490 | 0.0950 | 0.1248 | 0.1156 | 0.2195 |
| `qwen2.5-3b-sft__tasks-item2index-seqrec-fusionseqrec__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11__rid-20260211_055856` | ep10 (experiment note) | item2index+seqrec+fusionseqrec | 13660 | 0.0911 | 0.1180 | 0.1117 | 0.2131 |
| `qwen2.5-3b-sft__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11` | ep10 (experiment note) | seqrec+item2index (experiment note) | 4921 | 0.0667 | 0.0941 | 0.0852 | 0.1793 |

### 本节结论

- 同为 `cb64` 时，不同训练配方的波动显著，说明“训练框架/任务组合”对指标影响很大。
- 该组 `NDCG@10` 的 best-worst 差值为 `+0.0283`。

## GenRec: Industrial_and_Scientific（仅组内比较）

| Run | Checkpoint | Stage | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | --- | --- | --- | --- | --- |
| `Industrial_and_Scientific-qwen2.5-3b-instruct-grpo` | `checkpoint-1485` | sft->rl(grpo) | 0.1035 | 0.1381 | 0.1211 | 0.2191 |
| `Industrial_and_Scientific-qwen2.5-3b-instruct-grpo` | `checkpoint-1648` | sft->rl(grpo) | 0.1032 | 0.1379 | 0.1208 | 0.2186 |
| `Industrial_and_Scientific-qwen2.5-3b-instruct-grpo` | `checkpoint-495` | sft->rl(grpo) | 0.1008 | 0.1405 | 0.1191 | 0.2255 |
| `Industrial_and_Scientific-sft-dsz0-4gpu-eq8` | `checkpoint-260` | sft | 0.0956 | 0.1363 | 0.1147 | 0.2241 |
| `Industrial_and_Scientific-sft-dsz0-4gpu-eq8` | `checkpoint-320` | sft | 0.0838 | 0.1193 | 0.1025 | 0.2052 |

### 本节结论

- `best(GRPO) - best(SFT)` 为 `NDCG@10 +0.0079`，`HR@10 +0.0018`，其中 RL=`checkpoint-1485`，SFT=`checkpoint-260`。

## GenRec: Instruments（单独列出，不与 Industrial 混比）

| Run | Checkpoint | Stage | Config Note | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-500` | sft | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.1220 | 0.1626 | 0.1422 | 0.2548 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-550` | sft | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.1196 | 0.1520 | 0.1371 | 0.2331 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-300` | sft | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.1158 | 0.1533 | 0.1343 | 0.2378 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-250` | sft | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.1051 | 0.1463 | 0.1243 | 0.2345 |
| `Instruments-grec-sft-qwen4B-4-512-dsz0` | `checkpoint-405` | sft | cb4-512, qwen3-4B emb, split=grec(leave-2-out) | 0.0955 | 0.1198 | 0.1122 | 0.1972 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-1665` | sft->rl(grpo) | cb4-256, qwen3-4B emb, split=grec(leave-2-out), SFT->RL(task1+task4+task5), init=from-sft495 | 0.0952 | 0.1145 | 0.1071 | 0.1696 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-2664` | sft->rl(grpo) | cb4-256, qwen3-4B emb, split=grec(leave-2-out), SFT->RL(task1+task4+task5), init=from-sft495 | 0.0946 | 0.1120 | 0.1055 | 0.1618 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-999` | sft->rl(grpo) | cb4-256, qwen3-4B emb, split=grec(leave-2-out), SFT->RL(task1+task4+task5), init=from-sft495 | 0.0938 | 0.1128 | 0.1067 | 0.1723 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-2331` | sft->rl(grpo) | cb4-256, qwen3-4B emb, split=grec(leave-2-out), SFT->RL(task1+task4+task5), init=from-sft495 | 0.0936 | 0.1099 | 0.1046 | 0.1606 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-1998` | sft->rl(grpo) | cb4-256, qwen3-4B emb, split=grec(leave-2-out), SFT->RL(task1+task4+task5), init=from-sft495 | 0.0934 | 0.1104 | 0.1041 | 0.1597 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-1332` | sft->rl(grpo) | cb4-256, qwen3-4B emb, split=grec(leave-2-out), SFT->RL(task1+task4+task5), init=from-sft495 | 0.0928 | 0.1107 | 0.1049 | 0.1665 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-200` | sft | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.0913 | 0.1331 | 0.1084 | 0.2131 |
| `Instruments-grec-sft-qwen4B-4-512-dsz0` | `checkpoint-450` | sft | cb4-512, qwen3-4B emb, split=grec(leave-2-out) | 0.0905 | 0.1114 | 0.1054 | 0.1801 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-333` | sft->rl(grpo) | cb4-256, qwen3-4B emb, split=grec(leave-2-out), SFT->RL(task1+task4+task5), init=from-sft495 | 0.0904 | 0.1108 | 0.1038 | 0.1726 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-666` | sft->rl(grpo) | cb4-256, qwen3-4B emb, split=grec(leave-2-out), SFT->RL(task1+task4+task5), init=from-sft495 | 0.0896 | 0.1082 | 0.1029 | 0.1694 |
| `Instruments-grec-sft-qwen4B-4-256-dsz0` | `checkpoint-495` | sft | cb4-256, qwen3-4B emb, split=grec(leave-2-out) | 0.0823 | 0.1094 | 0.0985 | 0.1844 |
| `Instruments-grec-sft-qwen4B-4-32-dsz0` | `checkpoint-765` | sft | cb4-32, qwen3-4B emb, split=grec(leave-2-out) | 0.0727 | 0.0968 | 0.0884 | 0.1699 |
| `Instruments-grec-sft-qwen4B-4-256-dsz0` | `checkpoint-630` | sft | cb4-256, qwen3-4B emb, split=grec(leave-2-out) | 0.0706 | 0.0964 | 0.0850 | 0.1635 |
| `Instruments-grec-sft-qwen4B-4-32-dsz0` | `checkpoint-890` | sft | cb4-32, qwen3-4B emb, split=grec(leave-2-out) | 0.0704 | 0.0908 | 0.0845 | 0.1565 |
| `Instruments-grec-sft-qwen4B-4-64-dsz0` | `checkpoint-675` | sft | cb4-64, qwen3-4B emb, split=grec(leave-2-out) | 0.0630 | 0.0978 | 0.0788 | 0.1707 |
| `Instruments-grec-sft-qwen4B-4-128-dsz0` | `checkpoint-405` | sft | cb4-128, qwen3-4B emb, split=grec(leave-2-out) | 0.0628 | 0.0899 | 0.0805 | 0.1713 |
| `Instruments-grec-sft-qwen4B-4-128-dsz0` | `checkpoint-360` | sft | cb4-128, qwen3-4B emb, split=grec(leave-2-out) | 0.0625 | 0.0910 | 0.0795 | 0.1690 |
| `Instruments-grec-sft-qwen4B-4-64-dsz0` | `checkpoint-810` | sft | cb4-64, qwen3-4B emb, split=grec(leave-2-out) | 0.0614 | 0.0929 | 0.0753 | 0.1565 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-100` | sft | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.0590 | 0.1108 | 0.0644 | 0.1339 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | `checkpoint-50` | sft | cb4-256, qwen3-4B emb, split=mimionerec(global 8:1:1) | 0.0231 | 0.0307 | 0.0270 | 0.0482 |

### 本节结论

- GenRec Instruments 当前最优为 `Instruments-mimionerec-sft-qwen4B-4-256-dsz0/checkpoint-500` (`NDCG@10=0.1220`, `HR@10=0.1626`)。
- 按当前记录的 best 对比，`best(RL)-best(SFT)` 为 `NDCG@10 -0.0268`，`HR@10 -0.0481`。

## GenRec: Instruments-grec SFT Codebook Sweep

| Codebook | Run | Best Checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 | ΔNDCG@10 vs cb32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | `Instruments-grec-sft-qwen4B-4-32-dsz0` | `checkpoint-765` | 0.0727 | 0.0968 | 0.0884 | 0.1699 | +0.0000 |
| 64 | `Instruments-grec-sft-qwen4B-4-64-dsz0` | `checkpoint-675` | 0.0630 | 0.0978 | 0.0788 | 0.1707 | -0.0097 |
| 128 | `Instruments-grec-sft-qwen4B-4-128-dsz0` | `checkpoint-405` | 0.0628 | 0.0899 | 0.0805 | 0.1713 | -0.0099 |
| 256 | `Instruments-grec-sft-qwen4B-4-256-dsz0` | `checkpoint-495` | 0.0823 | 0.1094 | 0.0985 | 0.1844 | +0.0096 |
| 512 | `Instruments-grec-sft-qwen4B-4-512-dsz0` | `checkpoint-405` | 0.0955 | 0.1198 | 0.1122 | 0.1972 | +0.0228 |

### 本节结论

- 最优 codebook 组为 `Instruments-grec-sft-qwen4B-4-512-dsz0/checkpoint-405` (`NDCG@10=0.0955`, `HR@10=0.1198`)。
- 最优与最弱 codebook 组的 `NDCG@10` 差值为 `+0.0327`。

## GenRec: Instruments-grec RL 轨迹（GRPO，含 checkpoint-0 基线）

| Run | Checkpoint | Step | NDCG@10 | HR@10 | NDCG@50 | HR@50 | ΔNDCG@10 vs prev | ΔHR@10 vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-0(from-sft495)` | 0 | 0.0823 | 0.1094 | 0.0985 | 0.1844 | - | - |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-333` | 333 | 0.0904 | 0.1108 | 0.1038 | 0.1726 | +0.0081 | +0.0014 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-666` | 666 | 0.0896 | 0.1082 | 0.1029 | 0.1694 | -0.0008 | -0.0026 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-999` | 999 | 0.0938 | 0.1128 | 0.1067 | 0.1723 | +0.0042 | +0.0046 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-1332` | 1332 | 0.0928 | 0.1107 | 0.1049 | 0.1665 | -0.0010 | -0.0021 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-1665` | 1665 | 0.0952 | 0.1145 | 0.1071 | 0.1696 | +0.0024 | +0.0038 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-1998` | 1998 | 0.0934 | 0.1104 | 0.1041 | 0.1597 | -0.0018 | -0.0041 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-2331` | 2331 | 0.0936 | 0.1099 | 0.1046 | 0.1606 | +0.0002 | -0.0005 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | `checkpoint-2664` | 2664 | 0.0946 | 0.1120 | 0.1055 | 0.1618 | +0.0010 | +0.0021 |

### 本节结论

- RL 最优 checkpoint: `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495/checkpoint-1665` (`NDCG@10=0.0952`, `HR@10=0.1145`)。
- `best(RL) - best(grec SFT)`：`NDCG@10 -0.0003`，`HR@10 -0.0053`。
- 对齐初始化基线 `from-sft495` 后：`NDCG@10 +0.0129`，`HR@10 +0.0051`。

## 训练框架 × Codebook 一致性分析（Instruments）

### 对比口径

- 框架 A: `GRec instruct-sft + tasks-item2index-seqrec-fusionseqrec`（RQ4 cb sweep）。
- 框架 B: `GenRec Instruments-grec SFT`。
- 框架 C: `GenRec Instruments-grec RL (GRPO)`（当前仅 cb256）。
- 比较方式: 各框架在每个 cb 上取 `NDCG@10` 最优 checkpoint。

### 共享 cb 对照（A vs B）

| Codebook | A Best Ckpt | A NDCG@10 | A HR@10 | B Best Ckpt | B NDCG@10 | B HR@10 | A-B NDCG@10 | A-B HR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | `checkpoint-13660` | 0.0820 | 0.1101 | `checkpoint-765` | 0.0727 | 0.0968 | +0.0093 | +0.0133 |
| 64 | `checkpoint-20490` | 0.0950 | 0.1248 | `checkpoint-675` | 0.0630 | 0.0978 | +0.0320 | +0.0270 |
| 128 | `checkpoint-13660` | 0.0833 | 0.1182 | `checkpoint-405` | 0.0628 | 0.0899 | +0.0205 | +0.0283 |
| 256 | `checkpoint-13660` | 0.0919 | 0.1266 | `checkpoint-495` | 0.0823 | 0.1094 | +0.0096 | +0.0172 |
| 512 | `checkpoint-13660` | 0.0940 | 0.1258 | `checkpoint-405` | 0.0955 | 0.1198 | -0.0015 | +0.0060 |

### cb256 RL 补充（框架 C）

| Framework | Best Checkpoint | NDCG@10 | HR@10 |
| --- | --- | --- | --- |
| A (GRec instruct-sft) | `checkpoint-13660` | 0.0919 | 0.1266 |
| B (GenRec grec-sft) | `checkpoint-495` | 0.0823 | 0.1094 |
| C (GenRec grec-rl) | `checkpoint-1665` | 0.0952 | 0.1145 |

### 本节结论

- 共享 cb 为 `32, 64, 128, 256, 512`；按 `NDCG@10` 的排序分别为 `A: cb64 > cb512 > cb256 > cb128 > cb32`，`B: cb512 > cb256 > cb32 > cb64 > cb128`。
- 排序 Spearman 相关系数约为 `0.200`。
- cb 结论仅弱一致，存在明显框架依赖差异（结论不完全一致）。
- 在 `cb256` 上，`C(best) - B(best)` 为 `NDCG@10 +0.0129`，`HR@10 +0.0051`。
- 在 `cb256` 上，`C(best) - A(best)` 为 `NDCG@10 +0.0033`，`HR@10 -0.0121`。

## Instruments: grec vs mimionerec 深入对比（GenRec, cb256 SFT）

- 本节仅比较 `cb256` 的 SFT 对照组，避免 codebook 宽度影响。

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
| `grec(cb256)` | `checkpoint-495` | 0.0823 | 0.1094 | 0.0985 | 0.1844 | - | - |
| `grec(cb256)` | `checkpoint-630` | 0.0706 | 0.0964 | 0.0850 | 0.1635 | -0.0117 | -0.0130 |
| `mimionerec(cb256)` | `checkpoint-50` | 0.0231 | 0.0307 | 0.0270 | 0.0482 | - | - |
| `mimionerec(cb256)` | `checkpoint-100` | 0.0590 | 0.1108 | 0.0644 | 0.1339 | +0.0359 | +0.0801 |
| `mimionerec(cb256)` | `checkpoint-200` | 0.0913 | 0.1331 | 0.1084 | 0.2131 | +0.0323 | +0.0223 |
| `mimionerec(cb256)` | `checkpoint-250` | 0.1051 | 0.1463 | 0.1243 | 0.2345 | +0.0138 | +0.0132 |
| `mimionerec(cb256)` | `checkpoint-300` | 0.1158 | 0.1533 | 0.1343 | 0.2378 | +0.0107 | +0.0070 |
| `mimionerec(cb256)` | `checkpoint-500` | 0.1220 | 0.1626 | 0.1422 | 0.2548 | +0.0062 | +0.0093 |
| `mimionerec(cb256)` | `checkpoint-550` | 0.1196 | 0.1520 | 0.1371 | 0.2331 | -0.0024 | -0.0106 |

- `grec` 最近一次从 `checkpoint-495` 到 `checkpoint-630`：`NDCG@10 -0.0117`，`HR@10 -0.0130`。

### 本节结论

- `mimionerec(best) - grec(best)` 为 `NDCG@10 +0.0397`，`HR@10 +0.0532`。
- `grec(cb256)` 最近一次从 `checkpoint-495` 到 `checkpoint-630`：`NDCG@10 -0.0117`，`HR@10 -0.0130`。

## Instruments 交叉框架对比（仅作参考）

- GRec 最优（Instruments）: `GRec_public/results/test/seqrec-constrained/Instruments-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003/qwen2.5-3b-instruct-sft__tasks-item2index-seqrec-fusionseqrec__idx-index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11__rid-ep15_20260212_012019/checkpoint-20490/results.json`, NDCG@10=0.0950
- GenRec 最优（Instruments）: `GenRec/results/Instruments-mimionerec-sft-qwen4B-4-256-dsz0/checkpoint-500/metrics.json`, NDCG@10=0.1220
- 差值 (GRec - GenRec): `-0.0270`

### 本节结论

- 当前最优点对比下，`GRec - GenRec` 的 `NDCG@10` 差值为 `-0.0270`；但该值仅代表“各自最优配置”的差异，不代表严格同配方下的框架增益。
