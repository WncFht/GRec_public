#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

METRIC_COLUMNS = [
    "HR@1",
    "HR@3",
    "HR@5",
    "HR@10",
    "HR@20",
    "HR@50",
    "NDCG@1",
    "NDCG@3",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "NDCG@50",
]

CB64_GROUP = "Instruments-rq4_cb64-64-64-64_sk0.0-0.0-0.0-0.003"


@dataclass
class PathsConfig:
    workspace_root: Path
    grec_results_root: Path
    genrec_results_root: Path
    output_tsv: Path
    output_md: Path


def _default_paths() -> PathsConfig:
    script_path = Path(__file__).resolve()
    grec_repo_root = script_path.parents[2]
    workspace_root = grec_repo_root.parent
    today = date.today().isoformat()
    return PathsConfig(
        workspace_root=workspace_root,
        grec_results_root=grec_repo_root / "results/test/seqrec-constrained",
        genrec_results_root=workspace_root / "GenRec/results",
        output_tsv=grec_repo_root / f"docs/seqrec_aggregated_metrics_{today}.tsv",
        output_md=grec_repo_root / f"docs/seqrec_results_summary_{today}.md",
    )


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Aggregate GRec/GenRec seqrec metrics into TSV + markdown summary."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=defaults.workspace_root,
        help="Workspace root containing GRec_public and GenRec.",
    )
    parser.add_argument(
        "--grec-results-root",
        type=Path,
        default=defaults.grec_results_root,
        help="GRec results root (contains **/results.json).",
    )
    parser.add_argument(
        "--genrec-results-root",
        type=Path,
        default=defaults.genrec_results_root,
        help="GenRec results root (contains **/metrics.json).",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=defaults.output_tsv,
        help="Output aggregated TSV path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=defaults.output_md,
        help="Output markdown summary path.",
    )
    return parser.parse_args()


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def fmt_signed(v: float | None) -> str:
    return "-" if v is None else f"{v:+.4f}"


def safe_delta(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None:
        return None
    return cur - prev


def safe_ratio(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / prev


def to_rel_str(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def grec_metric_map(mean_results: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in mean_results.items():
        v = to_float(val)
        if v is None:
            continue
        lk = key.lower()
        if lk.startswith("hit@"):
            out[f"HR@{lk.split('@', 1)[1]}"] = v
        elif lk.startswith("ndcg@"):
            out[f"NDCG@{lk.split('@', 1)[1]}"] = v
    return out


def parse_tasks_from_model(model: str) -> str:
    match = re.search(r"__tasks-(.*?)__idx-", model)
    if not match:
        return ""
    return match.group(1)


def infer_dataset_scope(group: str) -> str:
    if group.startswith("Instruments"):
        return "Instruments"
    if group.startswith("Industrial_and_Scientific"):
        return "Industrial_and_Scientific"
    return "Unknown"


def infer_checkpoint_step(checkpoint: str) -> int | None:
    match = re.search(r"checkpoint-(\d+)", checkpoint)
    if not match:
        return None
    return int(match.group(1))


def infer_grec_epoch_hint(group: str, model: str) -> str:
    match = re.search(r"__rid-ep(\d+)_", model)
    if match:
        return f"ep{match.group(1)} (run id)"
    if group == CB64_GROUP and model.startswith("qwen2.5-3b-sft__"):
        return "ep10 (experiment note)"
    return "-"


def infer_grec_task_hint(group: str, model: str) -> str:
    tasks_tag = parse_tasks_from_model(model)
    if tasks_tag:
        return tasks_tag.replace("-", "+")
    if group == CB64_GROUP and model.startswith("qwen2.5-3b-sft__idx-"):
        return "seqrec+item2index (experiment note)"
    return "-"


def infer_grec_stage(model: str) -> str:
    lower = model.lower()
    if "ranking" in lower:
        return "ranking"
    if "-instruct-sft" in lower:
        return "instruct-sft"
    if "-sft" in lower:
        return "sft"
    return "-"


def infer_genrec_stage(group: str) -> str:
    lower = group.lower()
    if "grpo" in lower:
        return "sft->rl(grpo)"
    if "sft" in lower:
        return "sft"
    return "-"


def infer_cb_width_from_name(name: str) -> int | None:
    width_match = re.search(r"(?:^|-)4-(\d+)(?:-|$)", name)
    if width_match:
        return int(width_match.group(1))

    cb_match = re.search(r"cb(\d+)", name, re.IGNORECASE)
    if cb_match:
        return int(cb_match.group(1))

    return None


def infer_genrec_task_hint(group: str) -> str:
    if group.startswith("Industrial_and_Scientific"):
        if "grpo" in group.lower():
            return "SFT(task1+task2+task3)->RL(task1+task4+task5)"
        return "SFT(task1+task2+task3)"

    cb_width = infer_cb_width_from_name(group)
    cb_note = f"cb4-{cb_width}" if cb_width is not None else "cb=unknown"

    if group.startswith("Instruments-grec-grpo"):
        from_sft_match = re.search(r"from-sft(\d+)", group)
        from_sft_note = (
            f", init=from-sft{from_sft_match.group(1)}" if from_sft_match else ""
        )
        return (
            f"{cb_note}, qwen3-4B emb, split=grec(leave-2-out), "
            f"SFT->RL(task1+task4+task5){from_sft_note}"
        )

    if group.startswith("Instruments-grec-sft"):
        return f"{cb_note}, qwen3-4B emb, split=grec(leave-2-out)"

    if group.startswith("Instruments-mimionerec-sft"):
        return f"{cb_note}, qwen3-4B emb, split=mimionerec(global 8:1:1)"

    return "-"


def collect_grec_rows(grec_root: Path, workspace_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(grec_root.glob("**/results.json")):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mean_results = data.get("mean_results")
        if not isinstance(mean_results, dict):
            continue

        rel = result_path.relative_to(grec_root)
        parts = rel.parts
        group = parts[0] if len(parts) >= 1 else ""
        model = ""
        checkpoint = ""
        if len(parts) >= 4 and parts[-1] == "results.json":
            model = "/".join(parts[1:-2])
            checkpoint = parts[-2]
        elif len(parts) >= 3 and parts[-1] == "results.json":
            checkpoint = parts[-2]

        metrics = grec_metric_map(mean_results)
        row: dict[str, Any] = {
            "source": "GRec",
            "dataset_scope": infer_dataset_scope(group),
            "group": group,
            "model": model,
            "checkpoint": checkpoint,
            "checkpoint_step": infer_checkpoint_step(checkpoint),
            "train_stage": infer_grec_stage(model),
            "tasks_hint": infer_grec_task_hint(group, model),
            "epoch_hint": infer_grec_epoch_hint(group, model),
            "path": to_rel_str(result_path, workspace_root),
        }
        for key in METRIC_COLUMNS:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def collect_genrec_rows(
    genrec_root: Path, workspace_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(genrec_root.glob("**/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        rel = metrics_path.relative_to(genrec_root)
        parts = rel.parts
        group = parts[0] if len(parts) >= 1 else ""
        checkpoint = parts[1] if len(parts) >= 2 else ""
        row: dict[str, Any] = {
            "source": "GenRec",
            "dataset_scope": infer_dataset_scope(group),
            "group": group,
            "model": "",
            "checkpoint": checkpoint,
            "checkpoint_step": infer_checkpoint_step(checkpoint),
            "train_stage": infer_genrec_stage(group),
            "tasks_hint": infer_genrec_task_hint(group),
            "epoch_hint": "-",
            "path": to_rel_str(metrics_path, workspace_root),
        }
        for key in METRIC_COLUMNS:
            row[key] = to_float(metrics.get(key))
        rows.append(row)
    return rows


def sort_by_ndcg10_desc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda x: x.get("NDCG@10") if x.get("NDCG@10") is not None else -1.0,
        reverse=True,
    )


def best_row_by_metric(
    rows: list[dict[str, Any]], metric: str = "NDCG@10"
) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda x: (
            x.get(metric) if x.get(metric) is not None else -1.0,
            x.get("checkpoint_step") if x.get("checkpoint_step") is not None else -1,
        ),
        reverse=True,
    )[0]


def best_rows_by_cb(
    rows: list[dict[str, Any]], metric: str = "NDCG@10"
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        cb = infer_cb_width_from_name(str(row.get("group", "")))
        if cb is None:
            continue
        val = row.get(metric)
        if val is None:
            continue
        prev = out.get(cb)
        if prev is None:
            out[cb] = row
            continue
        prev_val = prev.get(metric)
        prev_step = prev.get("checkpoint_step")
        cur_step = row.get("checkpoint_step")
        if prev_val is None or float(val) > float(prev_val):
            out[cb] = row
        elif float(val) == float(prev_val) and (cur_step or -1) > (prev_step or -1):
            out[cb] = row
    return out


def rank_cbs_by_metric(
    cb_rows: dict[int, dict[str, Any]], metric: str = "NDCG@10"
) -> list[int]:
    return sorted(
        cb_rows.keys(),
        key=lambda cb: (
            cb_rows[cb].get(metric) if cb_rows[cb].get(metric) is not None else -1.0
        ),
        reverse=True,
    )


def spearman_rank_correlation(order_a: list[int], order_b: list[int]) -> float | None:
    if len(order_a) < 2 or len(order_a) != len(order_b):
        return None
    if set(order_a) != set(order_b):
        return None

    rank_a = {cb: i + 1 for i, cb in enumerate(order_a)}
    rank_b = {cb: i + 1 for i, cb in enumerate(order_b)}
    d2_sum = sum((rank_a[cb] - rank_b[cb]) ** 2 for cb in rank_a)
    n = len(order_a)
    return 1 - (6 * d2_sum) / (n * (n * n - 1))


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "source",
        "dataset_scope",
        "group",
        "model",
        "checkpoint",
        "checkpoint_step",
        "train_stage",
        "tasks_hint",
        "epoch_hint",
        "path",
        *METRIC_COLUMNS,
    ]
    lines = ["\t".join(header)]
    for row in rows:
        values = [
            str(row.get("source", "")),
            str(row.get("dataset_scope", "")),
            str(row.get("group", "")),
            str(row.get("model", "")),
            str(row.get("checkpoint", "")),
            "" if row.get("checkpoint_step") is None else str(row["checkpoint_step"]),
            str(row.get("train_stage", "")),
            str(row.get("tasks_hint", "")),
            str(row.get("epoch_hint", "")),
            str(row.get("path", "")),
        ]
        for key in METRIC_COLUMNS:
            v = row.get(key)
            values.append("" if v is None else f"{v:.6f}")
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
) -> list[str]:
    header = "| " + " | ".join([name for name, _ in columns]) + " |"
    align = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for _, key in columns:
            value = row.get(key)
            if key in METRIC_COLUMNS:
                vals.append(fmt(value))
            elif key == "checkpoint_step":
                vals.append("-" if value is None else str(value))
            else:
                vals.append(str(value) if value is not None else "-")
        body.append("| " + " | ".join(vals) + " |")
    return [header, align, *body]


def append_conclusion(lines: list[str], points: list[str]) -> None:
    if not points:
        return
    lines.append("### 本节结论")
    lines.append("")
    for point in points:
        lines.append(f"- {point}")
    lines.append("")


def build_markdown(rows: list[dict[str, Any]], cfg: PathsConfig) -> str:
    grec_rows = [r for r in rows if r["source"] == "GRec"]
    genrec_rows = [r for r in rows if r["source"] == "GenRec"]
    grec_sorted = sort_by_ndcg10_desc(grec_rows)

    grec_instr = [r for r in grec_rows if r["dataset_scope"] == "Instruments"]
    genrec_instr = [r for r in genrec_rows if r["dataset_scope"] == "Instruments"]
    genrec_industry = [
        r for r in genrec_rows if r["dataset_scope"] == "Industrial_and_Scientific"
    ]
    genrec_instr_grec_sft = [
        r for r in genrec_instr if r["group"].startswith("Instruments-grec-sft")
    ]
    genrec_instr_mimionerec_sft = [
        r for r in genrec_instr if r["group"].startswith("Instruments-mimionerec-sft")
    ]
    genrec_instr_grec_rl = [
        r for r in genrec_instr if r["group"].startswith("Instruments-grec-grpo")
    ]
    genrec_instr_grec_cb256_sft = [
        r for r in genrec_instr_grec_sft if infer_cb_width_from_name(r["group"]) == 256
    ]

    rq4_rows = []
    for row in grec_instr:
        if not row["group"].startswith("Instruments-rq4_cb"):
            continue
        if (
            "qwen2.5-3b-instruct-sft__tasks-item2index-seqrec-fusionseqrec"
            not in row["model"]
        ):
            continue
        match = re.search(r"cb(\d+)-", row["group"])
        if not match:
            continue
        rq4_rows.append((int(match.group(1)), row))
    rq4_rows.sort(key=lambda x: x[0])
    cb32 = next((r for cb, r in rq4_rows if cb == 32), None)

    cb64_rows = [r for r in grec_rows if r["group"] == CB64_GROUP]
    cb64_rows = sort_by_ndcg10_desc(cb64_rows)

    ind_sft = best_row_by_metric(
        [r for r in genrec_industry if r["train_stage"] == "sft"], "NDCG@10"
    )
    ind_rl = best_row_by_metric(
        [r for r in genrec_industry if "rl" in r["train_stage"]], "NDCG@10"
    )

    best_grec_overall = grec_sorted[0] if grec_sorted else None
    best_grec_instr = sort_by_ndcg10_desc(grec_instr)[0] if grec_instr else None
    best_genrec_instr = sort_by_ndcg10_desc(genrec_instr)[0] if genrec_instr else None
    best_genrec_instr_sft = best_row_by_metric(
        [r for r in genrec_instr if r["train_stage"] == "sft"], "NDCG@10"
    )
    best_genrec_instr_rl = best_row_by_metric(
        [r for r in genrec_instr if "rl" in r["train_stage"]], "NDCG@10"
    )

    lines: list[str] = []
    lines.append("# SeqRec 结果汇总（GRec + GenRec）")
    lines.append("")
    lines.append(f"- 生成日期: {date.today().isoformat()}")
    lines.append(f"- GRec 数据源: `{cfg.grec_results_root}`（{len(grec_rows)} runs）")
    lines.append(
        f"- GenRec 数据源: `{cfg.genrec_results_root}`（{len(genrec_rows)} runs）"
    )
    lines.append(f"- 汇总表: `{to_rel_str(cfg.output_tsv, cfg.workspace_root)}`")
    lines.append("")

    lines.append("## 目录")
    lines.append("")
    for toc_item in [
        "总体结论（TL;DR）",
        "可比性说明（重要）",
        "GenRec 任务定义（task1~task5）",
        "GRec 总榜（按 NDCG@10）",
        "RQ4 Codebook Sweep（Instruments, instruct+tasks-item2index-seqrec-fusionseqrec）",
        "cb64 组内对比（带实验注释）",
        "GenRec: Industrial_and_Scientific（仅组内比较）",
        "GenRec: Instruments（单独列出，不与 Industrial 混比）",
        "GenRec: Instruments-grec SFT Codebook Sweep",
        "GenRec: Instruments-grec RL 轨迹（GRPO，含 checkpoint-0 基线）",
        "训练框架 × Codebook 一致性分析（Instruments）",
        "Instruments: grec vs mimionerec 深入对比（GenRec, cb256 SFT）",
        "Instruments 交叉框架对比（仅作参考）",
    ]:
        lines.append(f"1. {toc_item}")
    lines.append("")

    lines.append("## 自动生成脚本")
    lines.append("")
    lines.append(
        f"- 脚本: `{to_rel_str(Path(__file__).resolve(), cfg.workspace_root)}`"
    )
    lines.append("- 命令:")
    lines.append("```bash")
    lines.append(
        f"python3 {to_rel_str(Path(__file__).resolve(), cfg.workspace_root)} "
        f"--workspace-root {cfg.workspace_root}"
    )
    lines.append("```")
    lines.append("")

    lines.append("## 总体结论（TL;DR）")
    lines.append("")
    tldr_points: list[str] = []
    if best_grec_overall:
        tldr_points.append(
            "GRec 当前全表最优为 "
            f"`{best_grec_overall['group']}/{best_grec_overall['checkpoint']}` "
            f"(`NDCG@10={fmt(best_grec_overall.get('NDCG@10'))}`, "
            f"`HR@10={fmt(best_grec_overall.get('HR@10'))}`)。"
        )
    if best_genrec_instr:
        tldr_points.append(
            "GenRec 当前 Instruments 最优为 "
            f"`{best_genrec_instr['group']}/{best_genrec_instr['checkpoint']}` "
            f"(`NDCG@10={fmt(best_genrec_instr.get('NDCG@10'))}`, "
            f"`HR@10={fmt(best_genrec_instr.get('HR@10'))}`)。"
        )
    if (
        ind_sft
        and ind_rl
        and ind_sft.get("NDCG@10") is not None
        and ind_rl.get("NDCG@10") is not None
    ):
        tldr_points.append(
            "Industrial_and_Scientific 上，`best(GRPO)-best(SFT)` 为 "
            f"`NDCG@10 {fmt_signed(safe_delta(ind_rl.get('NDCG@10'), ind_sft.get('NDCG@10')))}`，"
            f"`HR@10 {fmt_signed(safe_delta(ind_rl.get('HR@10'), ind_sft.get('HR@10')))}`。"
        )
    append_conclusion(lines, tldr_points)

    lines.append("## 可比性说明（重要）")
    lines.append("")
    lines.append(
        "- `GenRec` 同时包含 `Industrial_and_Scientific` 与 `Instruments` 两个数据集结果，应先做组内比较，再做跨组参考。"
    )
    lines.append(
        "- `Industrial_and_Scientific-qwen2.5-3b-instruct-grpo` 可作为 `Industrial_and_Scientific-sft-dsz0-4gpu-eq8` 的 SFT 后续 RL（GRPO）结果对比。"
    )
    lines.append(
        "- `GenRec/Instruments-grec-sft-*` 与 `GenRec/Instruments-mimionerec-sft-*` 使用相同模型与训练超参框架，主要变量是数据切分策略。"
    )
    lines.append(
        "- `GenRec/scripts/prepare_category_from_inter_json.py` 中，`grec` 采用 per-user leave-2-out；`mimionerec` 先构造全量 next-item 样本再做全局 8:1:1 切分。"
    )
    lines.append(
        "- `GenRec/hope/*-evaluate.sh` 显示两组评测分别读取各自数据变体的 `test.json`，因此绝对值差异应视为“切分+训练”联合效果。"
    )
    lines.append(
        "- `cb64` 组内三条 run 不是同配方：第一条为 `ep15`；后两条按实验说明是 `ep10`，且任务配置不同（`seqrec+item2index+fusionseqrec` vs `seqrec+item2index`），所以 checkpoint step 不能直接横向对齐。"
    )
    lines.append(
        "- `GenRec` 任务构成来自 `GenRec/preprocess_data_sft_rl.py:474` 与 `GenRec/preprocess_data_sft_rl.py:540`：SFT 使用 `task1+task2+task3`，RL 使用 `task1+task4+task5`。"
    )
    lines.append("")
    append_conclusion(
        lines,
        [
            "跨框架比较时需要先按数据集分层，再按任务定义与 split 方式对齐，避免把不可比差异当作模型差异。",
            "Instruments 上的 `grec`/`mimionerec` 与 RL/SFT 结论都应视为“切分策略 + 训练目标”的联合结果。",
        ],
    )

    lines.append("## GenRec 任务定义（task1~task5）")
    lines.append("")
    lines.append(
        "- 来源: `GenRec/preprocess_data_sft_rl.py:479`, `GenRec/preprocess_data_sft_rl.py:497`, `GenRec/preprocess_data_sft_rl.py:502`, `GenRec/preprocess_data_sft_rl.py:508`, `GenRec/preprocess_data_sft_rl.py:517`"
    )
    lines.append("")
    lines.extend(
        markdown_table(
            [
                {
                    "task": "task1_sid_sft",
                    "definition": "顺序推荐主任务：输入历史交互序列（semantic IDs），预测下一个 semantic ID。",
                    "usage": "SFT + RL",
                    "ability": "seq_rec",
                },
                {
                    "task": "task2_sid_item_feat",
                    "definition": "item 属性问答：sid->title 与 title->sid 双向 QA。",
                    "usage": "SFT only",
                    "ability": "-",
                },
                {
                    "task": "task3_fusion_seq",
                    "definition": "FusionSeqRec：输入历史 semantic IDs，预测下一物品 title。",
                    "usage": "SFT only",
                    "ability": "-",
                },
                {
                    "task": "task4_hisTitle2sid",
                    "definition": "Title2Sid 序列任务：输入历史 item title 序列，预测下一项 semantic ID。",
                    "usage": "RL only",
                    "ability": "seq_title2sid",
                },
                {
                    "task": "task5_title_desc2sid",
                    "definition": "Title/Description2Sid：给 title 或 description，预测 semantic ID。",
                    "usage": "RL only",
                    "ability": "title_desc2sid",
                },
            ],
            [
                ("Task", "task"),
                ("Definition", "definition"),
                ("Used In", "usage"),
                ("RL Ability", "ability"),
            ],
        )
    )
    lines.append("")
    append_conclusion(
        lines,
        [
            "GenRec 的 SFT 与 RL 并非同一任务集合，RL 的收益应解读为 `task1+task4+task5` 下的能力重分配，而非纯粹延续 SFT 指标。"
        ],
    )

    lines.append("## GRec 总榜（按 NDCG@10）")
    lines.append("")
    grec_rank_rows = []
    for i, row in enumerate(grec_sorted, 1):
        grec_rank_rows.append(
            {
                "rank": i,
                "group": f"`{row['group']}`",
                "checkpoint": f"`{row['checkpoint']}`",
                "NDCG@10": row["NDCG@10"],
                "HR@10": row["HR@10"],
                "NDCG@50": row["NDCG@50"],
                "HR@50": row["HR@50"],
            }
        )
    lines.extend(
        markdown_table(
            grec_rank_rows,
            [
                ("Rank", "rank"),
                ("Group", "group"),
                ("Checkpoint", "checkpoint"),
                ("NDCG@10", "NDCG@10"),
                ("HR@10", "HR@10"),
                ("NDCG@50", "NDCG@50"),
                ("HR@50", "HR@50"),
            ],
        )
    )
    lines.append("")
    grec_rank_conclusion: list[str] = []
    if best_grec_overall:
        grec_rank_conclusion.append(
            "全量 GRec 结果中，"
            f"`{best_grec_overall['group']}/{best_grec_overall['checkpoint']}` "
            "当前为 NDCG@10 最优。"
        )
    if best_grec_instr:
        grec_rank_conclusion.append(
            "在 Instruments 子集里，"
            f"`{best_grec_instr['group']}/{best_grec_instr['checkpoint']}` "
            "是当前最优参考点。"
        )
    append_conclusion(lines, grec_rank_conclusion)

    if rq4_rows:
        lines.append(
            "## RQ4 Codebook Sweep（Instruments, instruct+tasks-item2index-seqrec-fusionseqrec）"
        )
        lines.append("")
        table_rows = []
        for cb, row in rq4_rows:
            delta = None
            if (
                cb32
                and row.get("NDCG@10") is not None
                and cb32.get("NDCG@10") is not None
            ):
                delta = row["NDCG@10"] - cb32["NDCG@10"]
            table_rows.append(
                {
                    "cb": cb,
                    "NDCG@10": row["NDCG@10"],
                    "HR@10": row["HR@10"],
                    "NDCG@50": row["NDCG@50"],
                    "HR@50": row["HR@50"],
                    "delta": "-" if delta is None else f"{delta:+.4f}",
                }
            )
        lines.extend(
            markdown_table(
                table_rows,
                [
                    ("Codebook", "cb"),
                    ("NDCG@10", "NDCG@10"),
                    ("HR@10", "HR@10"),
                    ("NDCG@50", "NDCG@50"),
                    ("HR@50", "HR@50"),
                    ("Delta vs cb32", "delta"),
                ],
            )
        )
        lines.append("")
        rq4_valid = [row for _, row in rq4_rows if row.get("NDCG@10") is not None]
        if rq4_valid:
            rq4_best = max(rq4_valid, key=lambda x: float(x["NDCG@10"]))
            rq4_worst = min(rq4_valid, key=lambda x: float(x["NDCG@10"]))
            append_conclusion(
                lines,
                [
                    "在当前 GRec RQ4 sweep 中，"
                    f"`{rq4_best['group']}/{rq4_best['checkpoint']}` 最优，"
                    f"`{rq4_worst['group']}/{rq4_worst['checkpoint']}` 最弱。",
                    "最优与最弱的 `NDCG@10` 差值为 "
                    f"`{fmt_signed(safe_delta(rq4_best.get('NDCG@10'), rq4_worst.get('NDCG@10')))}`。",
                ],
            )

    if cb64_rows:
        lines.append("## cb64 组内对比（带实验注释）")
        lines.append("")
        table_rows = []
        for row in cb64_rows:
            table_rows.append(
                {
                    "model": f"`{row['model']}`",
                    "epoch_hint": row["epoch_hint"],
                    "tasks_hint": row["tasks_hint"],
                    "checkpoint_step": row["checkpoint_step"],
                    "NDCG@10": row["NDCG@10"],
                    "HR@10": row["HR@10"],
                    "NDCG@50": row["NDCG@50"],
                    "HR@50": row["HR@50"],
                }
            )
        lines.extend(
            markdown_table(
                table_rows,
                [
                    ("Model Variant", "model"),
                    ("Epoch Hint", "epoch_hint"),
                    ("Task Hint", "tasks_hint"),
                    ("Step", "checkpoint_step"),
                    ("NDCG@10", "NDCG@10"),
                    ("HR@10", "HR@10"),
                    ("NDCG@50", "NDCG@50"),
                    ("HR@50", "HR@50"),
                ],
            )
        )
        lines.append("")
        cb64_valid = [row for row in cb64_rows if row.get("NDCG@10") is not None]
        if cb64_valid:
            cb64_best = cb64_valid[0]
            cb64_worst = cb64_valid[-1]
            append_conclusion(
                lines,
                [
                    "同为 `cb64` 时，不同训练配方的波动显著，说明“训练框架/任务组合”对指标影响很大。",
                    "该组 `NDCG@10` 的 best-worst 差值为 "
                    f"`{fmt_signed(safe_delta(cb64_best.get('NDCG@10'), cb64_worst.get('NDCG@10')))}`。",
                ],
            )

    if genrec_industry:
        lines.append("## GenRec: Industrial_and_Scientific（仅组内比较）")
        lines.append("")
        ind_rows = sort_by_ndcg10_desc(genrec_industry)
        table_rows = []
        for row in ind_rows:
            table_rows.append(
                {
                    "group": f"`{row['group']}`",
                    "checkpoint": f"`{row['checkpoint']}`",
                    "stage": row["train_stage"],
                    "NDCG@10": row["NDCG@10"],
                    "HR@10": row["HR@10"],
                    "NDCG@50": row["NDCG@50"],
                    "HR@50": row["HR@50"],
                }
            )
        lines.extend(
            markdown_table(
                table_rows,
                [
                    ("Run", "group"),
                    ("Checkpoint", "checkpoint"),
                    ("Stage", "stage"),
                    ("NDCG@10", "NDCG@10"),
                    ("HR@10", "HR@10"),
                    ("NDCG@50", "NDCG@50"),
                    ("HR@50", "HR@50"),
                ],
            )
        )
        lines.append("")
        industry_points: list[str] = []
        if (
            ind_sft
            and ind_rl
            and ind_sft.get("NDCG@10") is not None
            and ind_rl.get("NDCG@10") is not None
        ):
            ndcg_delta = ind_rl["NDCG@10"] - ind_sft["NDCG@10"]
            hr_delta = safe_delta(ind_rl.get("HR@10"), ind_sft.get("HR@10"))
            industry_points.append(
                f"`best(GRPO) - best(SFT)` 为 `NDCG@10 {ndcg_delta:+.4f}`，"
                f"`HR@10 {fmt_signed(hr_delta)}`，"
                f"其中 RL=`{ind_rl['checkpoint']}`，SFT=`{ind_sft['checkpoint']}`。"
            )
        append_conclusion(lines, industry_points)

    if genrec_instr:
        lines.append("## GenRec: Instruments（单独列出，不与 Industrial 混比）")
        lines.append("")
        table_rows = []
        for row in sort_by_ndcg10_desc(genrec_instr):
            table_rows.append(
                {
                    "group": f"`{row['group']}`",
                    "checkpoint": f"`{row['checkpoint']}`",
                    "stage": row["train_stage"],
                    "tasks_hint": row["tasks_hint"],
                    "NDCG@10": row["NDCG@10"],
                    "HR@10": row["HR@10"],
                    "NDCG@50": row["NDCG@50"],
                    "HR@50": row["HR@50"],
                }
            )
        lines.extend(
            markdown_table(
                table_rows,
                [
                    ("Run", "group"),
                    ("Checkpoint", "checkpoint"),
                    ("Stage", "stage"),
                    ("Config Note", "tasks_hint"),
                    ("NDCG@10", "NDCG@10"),
                    ("HR@10", "HR@10"),
                    ("NDCG@50", "NDCG@50"),
                    ("HR@50", "HR@50"),
                ],
            )
        )
        lines.append("")
        instruments_points: list[str] = []
        if best_genrec_instr:
            instruments_points.append(
                "GenRec Instruments 当前最优为 "
                f"`{best_genrec_instr['group']}/{best_genrec_instr['checkpoint']}` "
                f"(`NDCG@10={fmt(best_genrec_instr.get('NDCG@10'))}`, "
                f"`HR@10={fmt(best_genrec_instr.get('HR@10'))}`)。"
            )
        if best_genrec_instr_sft and best_genrec_instr_rl:
            instruments_points.append(
                "按当前记录的 best 对比，`best(RL)-best(SFT)` 为 "
                f"`NDCG@10 {fmt_signed(safe_delta(best_genrec_instr_rl.get('NDCG@10'), best_genrec_instr_sft.get('NDCG@10')))}`，"
                f"`HR@10 {fmt_signed(safe_delta(best_genrec_instr_rl.get('HR@10'), best_genrec_instr_sft.get('HR@10')))}`。"
            )
        append_conclusion(lines, instruments_points)

    if genrec_instr_grec_sft:
        lines.append("## GenRec: Instruments-grec SFT Codebook Sweep")
        lines.append("")
        grec_sft_best_by_group: dict[str, dict[str, Any]] = {}
        for row in genrec_instr_grec_sft:
            group_name = str(row["group"])
            prev = grec_sft_best_by_group.get(group_name)
            if prev is None:
                grec_sft_best_by_group[group_name] = row
                continue

            cur_metric = row.get("NDCG@10") if row.get("NDCG@10") is not None else -1.0
            prev_metric = (
                prev.get("NDCG@10") if prev.get("NDCG@10") is not None else -1.0
            )
            if cur_metric > prev_metric:
                grec_sft_best_by_group[group_name] = row

        cb32_row = next(
            (
                row
                for group_name, row in grec_sft_best_by_group.items()
                if infer_cb_width_from_name(group_name) == 32
            ),
            None,
        )

        sweep_rows = []
        for group_name, row in sorted(
            grec_sft_best_by_group.items(),
            key=lambda item: (
                infer_cb_width_from_name(item[0])
                if infer_cb_width_from_name(item[0]) is not None
                else 10**9,
                item[0],
            ),
        ):
            cb_width = infer_cb_width_from_name(group_name)
            sweep_rows.append(
                {
                    "cb": cb_width if cb_width is not None else "-",
                    "group": f"`{group_name}`",
                    "checkpoint": f"`{row['checkpoint']}`",
                    "NDCG@10": row["NDCG@10"],
                    "HR@10": row["HR@10"],
                    "NDCG@50": row["NDCG@50"],
                    "HR@50": row["HR@50"],
                    "delta_vs_cb32": (
                        "-"
                        if cb32_row is None
                        else fmt_signed(
                            safe_delta(row.get("NDCG@10"), cb32_row.get("NDCG@10"))
                        )
                    ),
                }
            )
        lines.extend(
            markdown_table(
                sweep_rows,
                [
                    ("Codebook", "cb"),
                    ("Run", "group"),
                    ("Best Checkpoint", "checkpoint"),
                    ("NDCG@10", "NDCG@10"),
                    ("HR@10", "HR@10"),
                    ("NDCG@50", "NDCG@50"),
                    ("HR@50", "HR@50"),
                    ("ΔNDCG@10 vs cb32", "delta_vs_cb32"),
                ],
            )
        )
        lines.append("")

        valid_sweep_rows = [
            row
            for row in grec_sft_best_by_group.values()
            if row.get("NDCG@10") is not None
        ]
        sft_sweep_points: list[str] = []
        if valid_sweep_rows:
            best_sweep = max(valid_sweep_rows, key=lambda x: float(x["NDCG@10"]))
            worst_sweep = min(valid_sweep_rows, key=lambda x: float(x["NDCG@10"]))
            sweep_gap = safe_delta(
                best_sweep.get("NDCG@10"), worst_sweep.get("NDCG@10")
            )
            sft_sweep_points.append(
                f"最优 codebook 组为 `{best_sweep['group']}/{best_sweep['checkpoint']}` "
                f"(`NDCG@10={fmt(best_sweep.get('NDCG@10'))}`, `HR@10={fmt(best_sweep.get('HR@10'))}`)。"
            )
            sft_sweep_points.append(
                f"最优与最弱 codebook 组的 `NDCG@10` 差值为 `{fmt_signed(sweep_gap)}`。"
            )
        append_conclusion(lines, sft_sweep_points)

    if genrec_instr_grec_rl:
        lines.append(
            "## GenRec: Instruments-grec RL 轨迹（GRPO，含 checkpoint-0 基线）"
        )
        lines.append("")

        traj_rows = []
        rl_groups: dict[str, list[dict[str, Any]]] = {}
        for row in genrec_instr_grec_rl:
            rl_groups.setdefault(str(row["group"]), []).append(row)

        for group_name in sorted(rl_groups.keys()):
            rl_ordered = sorted(
                rl_groups[group_name],
                key=lambda x: (
                    x["checkpoint_step"]
                    if x.get("checkpoint_step") is not None
                    else 10**12
                ),
            )

            prev_for_delta: dict[str, Any] | None = None
            from_sft_match = re.search(r"from-sft(\d+)", group_name)
            cb_width = infer_cb_width_from_name(group_name)
            if from_sft_match and cb_width is not None:
                from_sft_step = int(from_sft_match.group(1))
                base_sft_candidates = [
                    row
                    for row in genrec_instr_grec_sft
                    if infer_cb_width_from_name(str(row["group"])) == cb_width
                    and row.get("checkpoint_step") == from_sft_step
                ]
                base_sft = best_row_by_metric(base_sft_candidates, "NDCG@10")
                if base_sft is not None:
                    traj_rows.append(
                        {
                            "group": f"`{group_name}`",
                            "checkpoint": f"`checkpoint-0(from-sft{from_sft_step})`",
                            "step": 0,
                            "NDCG@10": base_sft.get("NDCG@10"),
                            "HR@10": base_sft.get("HR@10"),
                            "NDCG@50": base_sft.get("NDCG@50"),
                            "HR@50": base_sft.get("HR@50"),
                            "d_ndcg10": "-",
                            "d_hr10": "-",
                        }
                    )
                    prev_for_delta = base_sft

            for row in rl_ordered:
                traj_rows.append(
                    {
                        "group": f"`{row['group']}`",
                        "checkpoint": f"`{row['checkpoint']}`",
                        "step": row["checkpoint_step"],
                        "NDCG@10": row["NDCG@10"],
                        "HR@10": row["HR@10"],
                        "NDCG@50": row["NDCG@50"],
                        "HR@50": row["HR@50"],
                        "d_ndcg10": fmt_signed(
                            safe_delta(
                                row.get("NDCG@10"),
                                None
                                if prev_for_delta is None
                                else prev_for_delta.get("NDCG@10"),
                            )
                        ),
                        "d_hr10": fmt_signed(
                            safe_delta(
                                row.get("HR@10"),
                                None
                                if prev_for_delta is None
                                else prev_for_delta.get("HR@10"),
                            )
                        ),
                    }
                )
                prev_for_delta = row

        lines.extend(
            markdown_table(
                traj_rows,
                [
                    ("Run", "group"),
                    ("Checkpoint", "checkpoint"),
                    ("Step", "step"),
                    ("NDCG@10", "NDCG@10"),
                    ("HR@10", "HR@10"),
                    ("NDCG@50", "NDCG@50"),
                    ("HR@50", "HR@50"),
                    ("ΔNDCG@10 vs prev", "d_ndcg10"),
                    ("ΔHR@10 vs prev", "d_hr10"),
                ],
            )
        )
        lines.append("")

        best_grec_rl = best_row_by_metric(genrec_instr_grec_rl, "NDCG@10")
        best_grec_sft = best_row_by_metric(genrec_instr_grec_sft, "NDCG@10")
        rl_points: list[str] = []
        if best_grec_rl:
            rl_points.append(
                f"RL 最优 checkpoint: `{best_grec_rl['group']}/{best_grec_rl['checkpoint']}` "
                f"(`NDCG@10={fmt(best_grec_rl.get('NDCG@10'))}`, `HR@10={fmt(best_grec_rl.get('HR@10'))}`)。"
            )

        if (
            best_grec_rl
            and best_grec_sft
            and best_grec_rl.get("NDCG@10") is not None
            and best_grec_sft.get("NDCG@10") is not None
        ):
            rl_points.append(
                f"`best(RL) - best(grec SFT)`："
                f"`NDCG@10 {fmt_signed(safe_delta(best_grec_rl.get('NDCG@10'), best_grec_sft.get('NDCG@10')))}`，"
                f"`HR@10 {fmt_signed(safe_delta(best_grec_rl.get('HR@10'), best_grec_sft.get('HR@10')))}`。"
            )

        if best_grec_rl:
            from_sft_match = re.search(r"from-sft(\d+)", str(best_grec_rl["group"]))
            cb_width = infer_cb_width_from_name(str(best_grec_rl["group"]))
            if from_sft_match and cb_width is not None:
                from_sft_step = int(from_sft_match.group(1))
                base_sft_candidates = [
                    row
                    for row in genrec_instr_grec_sft
                    if infer_cb_width_from_name(str(row["group"])) == cb_width
                    and row.get("checkpoint_step") == from_sft_step
                ]
                base_sft = best_row_by_metric(base_sft_candidates, "NDCG@10")
                if (
                    base_sft
                    and best_grec_rl.get("NDCG@10") is not None
                    and base_sft.get("NDCG@10") is not None
                ):
                    rl_points.append(
                        f"对齐初始化基线 `from-sft{from_sft_step}` 后："
                        f"`NDCG@10 {fmt_signed(safe_delta(best_grec_rl.get('NDCG@10'), base_sft.get('NDCG@10')))}`，"
                        f"`HR@10 {fmt_signed(safe_delta(best_grec_rl.get('HR@10'), base_sft.get('HR@10')))}`。"
                    )
        append_conclusion(lines, rl_points)

    framework_a_rows = [
        r
        for r in grec_instr
        if r["group"].startswith("Instruments-rq4_cb")
        and "qwen2.5-3b-instruct-sft__tasks-item2index-seqrec-fusionseqrec"
        in r["model"]
    ]
    framework_b_rows = list(genrec_instr_grec_sft)
    framework_c_rows = list(genrec_instr_grec_rl)

    if framework_a_rows or framework_b_rows or framework_c_rows:
        lines.append("## 训练框架 × Codebook 一致性分析（Instruments）")
        lines.append("")
        lines.append("### 对比口径")
        lines.append("")
        lines.append(
            "- 框架 A: `GRec instruct-sft + tasks-item2index-seqrec-fusionseqrec`（RQ4 cb sweep）。"
        )
        lines.append("- 框架 B: `GenRec Instruments-grec SFT`。")
        lines.append("- 框架 C: `GenRec Instruments-grec RL (GRPO)`（当前仅 cb256）。")
        lines.append("- 比较方式: 各框架在每个 cb 上取 `NDCG@10` 最优 checkpoint。")
        lines.append("")

        framework_a_best_by_cb = best_rows_by_cb(framework_a_rows, "NDCG@10")
        framework_b_best_by_cb = best_rows_by_cb(framework_b_rows, "NDCG@10")
        shared_cbs = sorted(
            set(framework_a_best_by_cb.keys()) & set(framework_b_best_by_cb.keys())
        )

        if shared_cbs:
            lines.append("### 共享 cb 对照（A vs B）")
            lines.append("")
            compare_rows = []
            for cb in shared_cbs:
                row_a = framework_a_best_by_cb[cb]
                row_b = framework_b_best_by_cb[cb]
                compare_rows.append(
                    {
                        "cb": cb,
                        "a_ckpt": f"`{row_a['checkpoint']}`",
                        "a_ndcg10": fmt(row_a.get("NDCG@10")),
                        "a_hr10": fmt(row_a.get("HR@10")),
                        "b_ckpt": f"`{row_b['checkpoint']}`",
                        "b_ndcg10": fmt(row_b.get("NDCG@10")),
                        "b_hr10": fmt(row_b.get("HR@10")),
                        "d_ndcg10": fmt_signed(
                            safe_delta(row_a.get("NDCG@10"), row_b.get("NDCG@10"))
                        ),
                        "d_hr10": fmt_signed(
                            safe_delta(row_a.get("HR@10"), row_b.get("HR@10"))
                        ),
                    }
                )
            lines.extend(
                markdown_table(
                    compare_rows,
                    [
                        ("Codebook", "cb"),
                        ("A Best Ckpt", "a_ckpt"),
                        ("A NDCG@10", "a_ndcg10"),
                        ("A HR@10", "a_hr10"),
                        ("B Best Ckpt", "b_ckpt"),
                        ("B NDCG@10", "b_ndcg10"),
                        ("B HR@10", "b_hr10"),
                        ("A-B NDCG@10", "d_ndcg10"),
                        ("A-B HR@10", "d_hr10"),
                    ],
                )
            )
            lines.append("")

        if framework_c_rows:
            lines.append("### cb256 RL 补充（框架 C）")
            lines.append("")
            framework_c_cb256 = [
                r
                for r in framework_c_rows
                if infer_cb_width_from_name(str(r["group"])) == 256
            ]
            framework_c_best_cb256 = best_row_by_metric(framework_c_cb256, "NDCG@10")
            framework_b_cb256 = framework_b_best_by_cb.get(256)
            framework_a_cb256 = framework_a_best_by_cb.get(256)

            cb256_rows: list[dict[str, Any]] = []
            if framework_a_cb256:
                cb256_rows.append(
                    {
                        "framework": "A (GRec instruct-sft)",
                        "checkpoint": f"`{framework_a_cb256['checkpoint']}`",
                        "NDCG@10": framework_a_cb256.get("NDCG@10"),
                        "HR@10": framework_a_cb256.get("HR@10"),
                    }
                )
            if framework_b_cb256:
                cb256_rows.append(
                    {
                        "framework": "B (GenRec grec-sft)",
                        "checkpoint": f"`{framework_b_cb256['checkpoint']}`",
                        "NDCG@10": framework_b_cb256.get("NDCG@10"),
                        "HR@10": framework_b_cb256.get("HR@10"),
                    }
                )
            if framework_c_best_cb256:
                cb256_rows.append(
                    {
                        "framework": "C (GenRec grec-rl)",
                        "checkpoint": f"`{framework_c_best_cb256['checkpoint']}`",
                        "NDCG@10": framework_c_best_cb256.get("NDCG@10"),
                        "HR@10": framework_c_best_cb256.get("HR@10"),
                    }
                )

            if cb256_rows:
                lines.extend(
                    markdown_table(
                        cb256_rows,
                        [
                            ("Framework", "framework"),
                            ("Best Checkpoint", "checkpoint"),
                            ("NDCG@10", "NDCG@10"),
                            ("HR@10", "HR@10"),
                        ],
                    )
                )
                lines.append("")

        framework_points: list[str] = []
        if shared_cbs:
            order_a = rank_cbs_by_metric(
                {cb: framework_a_best_by_cb[cb] for cb in shared_cbs}, "NDCG@10"
            )
            order_b = rank_cbs_by_metric(
                {cb: framework_b_best_by_cb[cb] for cb in shared_cbs}, "NDCG@10"
            )
            rho = spearman_rank_correlation(order_a, order_b)
            framework_points.append(
                f"共享 cb 为 `{', '.join(str(cb) for cb in shared_cbs)}`；"
                f"按 `NDCG@10` 的排序分别为 "
                f"`A: {' > '.join(f'cb{cb}' for cb in order_a)}`，"
                f"`B: {' > '.join(f'cb{cb}' for cb in order_b)}`。"
            )
            if rho is not None:
                framework_points.append(f"排序 Spearman 相关系数约为 `{rho:.3f}`。")
                if rho >= 0.6:
                    framework_points.append("cb 结论整体较一致。")
                elif rho >= 0:
                    framework_points.append(
                        "cb 结论仅弱一致，存在明显框架依赖差异（结论不完全一致）。"
                    )
                else:
                    framework_points.append("cb 结论方向相反，跨框架不一致。")

        framework_c_best_cb256 = best_row_by_metric(
            [
                r
                for r in framework_c_rows
                if infer_cb_width_from_name(str(r["group"])) == 256
            ],
            "NDCG@10",
        )
        framework_b_cb256 = framework_b_best_by_cb.get(256)
        framework_a_cb256 = framework_a_best_by_cb.get(256)
        if framework_c_best_cb256 and framework_b_cb256:
            framework_points.append(
                "在 `cb256` 上，`C(best) - B(best)` 为 "
                f"`NDCG@10 {fmt_signed(safe_delta(framework_c_best_cb256.get('NDCG@10'), framework_b_cb256.get('NDCG@10')))}`，"
                f"`HR@10 {fmt_signed(safe_delta(framework_c_best_cb256.get('HR@10'), framework_b_cb256.get('HR@10')))}`。"
            )
        if framework_c_best_cb256 and framework_a_cb256:
            framework_points.append(
                "在 `cb256` 上，`C(best) - A(best)` 为 "
                f"`NDCG@10 {fmt_signed(safe_delta(framework_c_best_cb256.get('NDCG@10'), framework_a_cb256.get('NDCG@10')))}`，"
                f"`HR@10 {fmt_signed(safe_delta(framework_c_best_cb256.get('HR@10'), framework_a_cb256.get('HR@10')))}`。"
            )
        append_conclusion(lines, framework_points)

    if genrec_instr_grec_cb256_sft or genrec_instr_mimionerec_sft:
        lines.append("## Instruments: grec vs mimionerec 深入对比（GenRec, cb256 SFT）")
        lines.append("")
        lines.append("- 本节仅比较 `cb256` 的 SFT 对照组，避免 codebook 宽度影响。")
        lines.append("")
        lines.append("### 配置与数据构造差异")
        lines.append("")
        lines.extend(
            markdown_table(
                [
                    {
                        "variant": "`grec`",
                        "split": "per-user leave-2-out",
                        "preprocess": "`GenRec/scripts/run_instruments_preprocess_grec.sh`",
                        "train_yaml": "`GenRec/examples/train_full/Instruments/instruments_rec_full_sft_3b_dsz0_qwen4b_4_256_grec.yaml`",
                        "eval_script": "`GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-evaluate.sh`",
                    },
                    {
                        "variant": "`mimionerec`",
                        "split": "global next-item pool + ratio split (8:1:1)",
                        "preprocess": "`GenRec/scripts/run_instruments_preprocess_mimionerec.sh`",
                        "train_yaml": "`GenRec/examples/train_full/Instruments/instruments_rec_full_sft_3b_dsz0_qwen4b_4_256_mimionerec.yaml`",
                        "eval_script": "`GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-mimionerec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-mimionerec-evaluate.sh`",
                    },
                ],
                [
                    ("Variant", "variant"),
                    ("Split Strategy", "split"),
                    ("Preprocess", "preprocess"),
                    ("Train YAML", "train_yaml"),
                    ("Eval Script", "eval_script"),
                ],
            )
        )
        lines.append("")
        lines.append(
            "- 两份 YAML 的 backbone、batch size、LR、epoch、deepspeed 基本一致；核心实验变量是数据切分。"
        )
        lines.append("")
        gm_points: list[str] = []

        best_genrec_grec_instr = best_row_by_metric(
            genrec_instr_grec_cb256_sft, "NDCG@10"
        )
        best_genrec_mimionerec_instr = (
            best_row_by_metric(genrec_instr_mimionerec_sft, "NDCG@10")
            if genrec_instr_mimionerec_sft
            else None
        )

        if best_genrec_grec_instr or best_genrec_mimionerec_instr:
            lines.append("### 最优 checkpoint 对比")
            lines.append("")
            best_rows = []
            if best_genrec_grec_instr:
                best_rows.append(
                    {
                        "variant": "`grec`",
                        "checkpoint": f"`{best_genrec_grec_instr['checkpoint']}`",
                        "NDCG@10": best_genrec_grec_instr["NDCG@10"],
                        "HR@10": best_genrec_grec_instr["HR@10"],
                        "NDCG@50": best_genrec_grec_instr["NDCG@50"],
                        "HR@50": best_genrec_grec_instr["HR@50"],
                    }
                )
            if best_genrec_mimionerec_instr:
                best_rows.append(
                    {
                        "variant": "`mimionerec`",
                        "checkpoint": f"`{best_genrec_mimionerec_instr['checkpoint']}`",
                        "NDCG@10": best_genrec_mimionerec_instr["NDCG@10"],
                        "HR@10": best_genrec_mimionerec_instr["HR@10"],
                        "NDCG@50": best_genrec_mimionerec_instr["NDCG@50"],
                        "HR@50": best_genrec_mimionerec_instr["HR@50"],
                    }
                )
            lines.extend(
                markdown_table(
                    best_rows,
                    [
                        ("Variant", "variant"),
                        ("Best Checkpoint", "checkpoint"),
                        ("NDCG@10", "NDCG@10"),
                        ("HR@10", "HR@10"),
                        ("NDCG@50", "NDCG@50"),
                        ("HR@50", "HR@50"),
                    ],
                )
            )
            lines.append("")

            if best_genrec_grec_instr and best_genrec_mimionerec_instr:
                d_ndcg10 = safe_delta(
                    best_genrec_mimionerec_instr["NDCG@10"],
                    best_genrec_grec_instr["NDCG@10"],
                )
                d_hr10 = safe_delta(
                    best_genrec_mimionerec_instr["HR@10"],
                    best_genrec_grec_instr["HR@10"],
                )
                d_ndcg50 = safe_delta(
                    best_genrec_mimionerec_instr["NDCG@50"],
                    best_genrec_grec_instr["NDCG@50"],
                )
                d_hr50 = safe_delta(
                    best_genrec_mimionerec_instr["HR@50"],
                    best_genrec_grec_instr["HR@50"],
                )
                r_ndcg10 = safe_ratio(
                    best_genrec_mimionerec_instr["NDCG@10"],
                    best_genrec_grec_instr["NDCG@10"],
                )
                r_hr10 = safe_ratio(
                    best_genrec_mimionerec_instr["HR@10"],
                    best_genrec_grec_instr["HR@10"],
                )
                lines.append(
                    f"- `mimionerec(best) - grec(best)`: "
                    f"`NDCG@10 {fmt_signed(d_ndcg10)}` (`{fmt_signed(r_ndcg10 * 100 if r_ndcg10 is not None else None)}%`), "
                    f"`HR@10 {fmt_signed(d_hr10)}` (`{fmt_signed(r_hr10 * 100 if r_hr10 is not None else None)}%`), "
                    f"`NDCG@50 {fmt_signed(d_ndcg50)}`，`HR@50 {fmt_signed(d_hr50)}`。"
                )
                lines.append("")
                gm_points.append(
                    "`mimionerec(best) - grec(best)` 为 "
                    f"`NDCG@10 {fmt_signed(d_ndcg10)}`，`HR@10 {fmt_signed(d_hr10)}`。"
                )

        lines.append("### Checkpoint 轨迹")
        lines.append("")
        traj_rows = []
        for variant_name, variant_rows in [
            ("`grec(cb256)`", genrec_instr_grec_cb256_sft),
            ("`mimionerec(cb256)`", genrec_instr_mimionerec_sft),
        ]:
            ordered = sorted(
                variant_rows,
                key=lambda x: (
                    x["checkpoint_step"]
                    if x.get("checkpoint_step") is not None
                    else 10**12
                ),
            )
            prev = None
            for row in ordered:
                d_ndcg10 = safe_delta(
                    row.get("NDCG@10"),
                    None if prev is None else prev.get("NDCG@10"),
                )
                d_hr10 = safe_delta(
                    row.get("HR@10"),
                    None if prev is None else prev.get("HR@10"),
                )
                traj_rows.append(
                    {
                        "variant": variant_name,
                        "checkpoint": f"`{row['checkpoint']}`",
                        "NDCG@10": row["NDCG@10"],
                        "HR@10": row["HR@10"],
                        "NDCG@50": row["NDCG@50"],
                        "HR@50": row["HR@50"],
                        "d_ndcg10": fmt_signed(d_ndcg10),
                        "d_hr10": fmt_signed(d_hr10),
                    }
                )
                prev = row
        if traj_rows:
            lines.extend(
                markdown_table(
                    traj_rows,
                    [
                        ("Variant", "variant"),
                        ("Checkpoint", "checkpoint"),
                        ("NDCG@10", "NDCG@10"),
                        ("HR@10", "HR@10"),
                        ("NDCG@50", "NDCG@50"),
                        ("HR@50", "HR@50"),
                        ("ΔNDCG@10 vs prev", "d_ndcg10"),
                        ("ΔHR@10 vs prev", "d_hr10"),
                    ],
                )
            )
            lines.append("")

        if genrec_instr_grec_cb256_sft:
            grec_ordered = sorted(
                genrec_instr_grec_cb256_sft,
                key=lambda x: (
                    x["checkpoint_step"]
                    if x.get("checkpoint_step") is not None
                    else 10**12
                ),
            )
            if len(grec_ordered) >= 2:
                g_last = grec_ordered[-1]
                g_prev = grec_ordered[-2]
                d_last_ndcg10 = safe_delta(g_last.get("NDCG@10"), g_prev.get("NDCG@10"))
                d_last_hr10 = safe_delta(g_last.get("HR@10"), g_prev.get("HR@10"))
                lines.append(
                    f"- `grec` 最近一次从 `{g_prev['checkpoint']}` 到 `{g_last['checkpoint']}`："
                    f"`NDCG@10 {fmt_signed(d_last_ndcg10)}`，`HR@10 {fmt_signed(d_last_hr10)}`。"
                )
                gm_points.append(
                    f"`grec(cb256)` 最近一次从 `{g_prev['checkpoint']}` 到 `{g_last['checkpoint']}`："
                    f"`NDCG@10 {fmt_signed(d_last_ndcg10)}`，`HR@10 {fmt_signed(d_last_hr10)}`。"
                )
        if genrec_instr_mimionerec_sft:
            mimi_ordered = sorted(
                genrec_instr_mimionerec_sft,
                key=lambda x: (
                    x["checkpoint_step"]
                    if x.get("checkpoint_step") is not None
                    else 10**12
                ),
            )
            if len(mimi_ordered) >= 2:
                all_up_ndcg10 = True
                all_up_hr10 = True
                for i in range(1, len(mimi_ordered)):
                    if (
                        safe_delta(
                            mimi_ordered[i].get("NDCG@10"),
                            mimi_ordered[i - 1].get("NDCG@10"),
                        )
                        is not None
                        and safe_delta(
                            mimi_ordered[i].get("NDCG@10"),
                            mimi_ordered[i - 1].get("NDCG@10"),
                        )
                        < 0
                    ):
                        all_up_ndcg10 = False
                    if (
                        safe_delta(
                            mimi_ordered[i].get("HR@10"),
                            mimi_ordered[i - 1].get("HR@10"),
                        )
                        is not None
                        and safe_delta(
                            mimi_ordered[i].get("HR@10"),
                            mimi_ordered[i - 1].get("HR@10"),
                        )
                        < 0
                    ):
                        all_up_hr10 = False
                if all_up_ndcg10 and all_up_hr10:
                    lines.append(
                        "- `mimionerec` 当前已记录 checkpoints 上 `NDCG@10` 与 `HR@10` 呈单调上升。"
                    )
                    gm_points.append(
                        "`mimionerec(cb256)` 在当前已记录 checkpoints 上，`NDCG@10` 与 `HR@10` 呈单调上升。"
                    )
        lines.append("")
        append_conclusion(lines, gm_points)

    if best_grec_instr and best_genrec_instr:
        lines.append("## Instruments 交叉框架对比（仅作参考）")
        lines.append("")
        delta = best_grec_instr["NDCG@10"] - best_genrec_instr["NDCG@10"]
        lines.append(
            f"- GRec 最优（Instruments）: `{best_grec_instr['path']}`, NDCG@10={fmt(best_grec_instr['NDCG@10'])}"
        )
        lines.append(
            f"- GenRec 最优（Instruments）: `{best_genrec_instr['path']}`, NDCG@10={fmt(best_genrec_instr['NDCG@10'])}"
        )
        lines.append(f"- 差值 (GRec - GenRec): `{delta:+.4f}`")
        lines.append("")
        append_conclusion(
            lines,
            [
                "当前最优点对比下，"
                f"`GRec - GenRec` 的 `NDCG@10` 差值为 `{delta:+.4f}`；"
                "但该值仅代表“各自最优配置”的差异，不代表严格同配方下的框架增益。"
            ],
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = PathsConfig(
        workspace_root=args.workspace_root.resolve(),
        grec_results_root=args.grec_results_root.resolve(),
        genrec_results_root=args.genrec_results_root.resolve(),
        output_tsv=args.output_tsv.resolve(),
        output_md=args.output_md.resolve(),
    )

    if not cfg.grec_results_root.exists():
        raise SystemExit(f"GRec results root not found: {cfg.grec_results_root}")
    if not cfg.genrec_results_root.exists():
        raise SystemExit(f"GenRec results root not found: {cfg.genrec_results_root}")

    grec_rows = collect_grec_rows(cfg.grec_results_root, cfg.workspace_root)
    genrec_rows = collect_genrec_rows(cfg.genrec_results_root, cfg.workspace_root)
    rows = grec_rows + genrec_rows

    write_tsv(rows, cfg.output_tsv)
    cfg.output_md.parent.mkdir(parents=True, exist_ok=True)
    summary = build_markdown(rows, cfg)
    cfg.output_md.write_text(summary, encoding="utf-8")

    print(f"Wrote TSV: {cfg.output_tsv}")
    print(f"Wrote MD : {cfg.output_md}")
    print(f"Rows: GRec={len(grec_rows)}, GenRec={len(genrec_rows)}, Total={len(rows)}")


if __name__ == "__main__":
    main()
