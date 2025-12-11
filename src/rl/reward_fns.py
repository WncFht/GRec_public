from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import repeat
from typing import Any


@dataclass
class _RewardContext:
    num_generations: int
    ndcg_rewards: list[float]
    pad_token_id: int | None = None
    pad_token: str | None = None


_REWARD_CONTEXT: _RewardContext | None = None
_SEQREC_PATTERN: re.Pattern[str] | None = None
_PRM_TOKEN_LIMIT = 4


def _build_seqrec_pattern(pad_token: str | None) -> re.Pattern[str]:
    """
    Build a regex that accepts the canonical seqrec format and ignores trailing pad tokens.
    """
    base = r"^<a_[^<>\n]+><b_[^<>\n]+><c_[^<>\n]+><d_[^<>\n]+><\|im_end\|>"
    if pad_token:
        escaped_pad = re.escape(pad_token.strip())
        base += f"(?:{escaped_pad})*"
    base += "$"
    return re.compile(base)


def _extract_gt_tokens(rm: dict[str, Any]) -> list[int]:
    gt = rm.get("ground_truth", "")
    if isinstance(gt, dict):
        tokens = gt.get("token", []) or []
        if tokens:
            return tokens
    msg = "ground_truth must include token ids for reward computation."
    raise RuntimeError(msg)


def initialize_reward_functions(
    num_generations: int,
    pad_token_id: int | None = None,
    pad_token: str | None = None,
) -> bool:
    """Prepare reward helpers and optionally run a quick sanity check."""
    global _REWARD_CONTEXT, _SEQREC_PATTERN
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    ndcg_rewards = [-elm / sum(ndcg_rewards) for elm in ndcg_rewards]
    _SEQREC_PATTERN = _build_seqrec_pattern(pad_token)
    _REWARD_CONTEXT = _RewardContext(
        num_generations, ndcg_rewards, pad_token_id=pad_token_id, pad_token=pad_token
    )

    return False


def _strip_padding_tokens(tokens: Iterable[int], pad_token_id: int | None) -> list[int]:
    if pad_token_id is None:
        return list(tokens)
    return [tok for tok in tokens if tok != pad_token_id]


def _prm_compare_length(
    tokens: list[int], gt_tokens: list[int], limit: int = _PRM_TOKEN_LIMIT
) -> int:
    if not tokens or not gt_tokens or limit <= 0:
        return 0
    return min(limit, len(tokens), len(gt_tokens))


def ndcg_rule_reward(
    reward_model: Iterable[dict[str, Any]],
    completions: Iterable[list[dict[str, str]]],
    completion_token_ids: Iterable[list[int]] | None = None,
    prompts=None,
    data_source: str | None = None,
    use_prm: bool = False,
    **unused,
):
    ctx = _ensure_context()
    repeat = ctx.num_generations
    pad_token_id = ctx.pad_token_id
    rewards: list[float] | list[list[float]] = []
    flag = False
    lis: list[float] = []
    group_records: list[dict[str, Any]] = []

    format_rewards = format_reward(
        completions,
        completion_token_ids=completion_token_ids,
        prompts=prompts,
        data_source=data_source,
    )
    if completion_token_ids is None:
        raise RuntimeError("completion_token_ids must be provided for token matching.")

    if use_prm:
        for i, (tokens, rm, fr) in enumerate(
            zip(completion_token_ids, reward_model, format_rewards, strict=False)
        ):
            gt_tokens = _strip_padding_tokens(_extract_gt_tokens(rm), pad_token_id)
            norm_tokens = _strip_padding_tokens(tokens, pad_token_id)
            group_records.append(
                {
                    "norm_tokens": norm_tokens,
                    "gt_tokens": gt_tokens,
                    "format_reward": float(fr),
                    "is_correct": fr > 0 and norm_tokens == gt_tokens,
                    "ndcg_value": ctx.ndcg_rewards[i % repeat],
                }
            )

            is_group_end = (i + 1) % repeat == 0
            is_last_item = i == len(format_rewards) - 1
            if not (is_group_end or is_last_item):
                continue

            any_correct = any(rec["is_correct"] for rec in group_records)
            for rec in group_records:
                fr_val: float = rec["format_reward"]
                norm_tokens = rec["norm_tokens"]
                gt_tokens = rec["gt_tokens"]
                if fr_val <= 0:
                    length = max(len(norm_tokens), 1)
                    rewards.append([fr_val] * length)
                    continue

                compare_len = _prm_compare_length(norm_tokens, gt_tokens)
                length = max(len(norm_tokens), compare_len)
                base_reward = (
                    0.0 if rec["is_correct"] or not any_correct else rec["ndcg_value"]
                )
                token_rewards = [0.0] * length
                for idx in range(compare_len):
                    if norm_tokens[idx] != gt_tokens[idx]:
                        token_rewards[idx] = base_reward
                rewards.append(token_rewards)

            group_records = []
        return rewards

    for i, (tokens, rm, fr) in enumerate(
        zip(completion_token_ids, reward_model, format_rewards, strict=False)
    ):
        gt_tokens = _strip_padding_tokens(_extract_gt_tokens(rm), pad_token_id)
        norm_tokens = _strip_padding_tokens(tokens, pad_token_id)
        if norm_tokens == gt_tokens and fr > 0:
            flag = True
            lis.append(0.0)
        else:
            lis.append(ctx.ndcg_rewards[i % ctx.num_generations])

        if (i + 1) % ctx.num_generations == 0:
            rewards.extend(lis if flag else [0.0] * repeat)
            flag = False
            lis = []

    return rewards


def rule_reward(
    reward_model: Iterable[dict[str, Any]],
    completions: Iterable[list[dict[str, str]]],
    completion_token_ids: Iterable[list[int]] | None = None,
    prompts=None,
    data_source: str | None = None,
    use_prm: bool = False,
    **unused,
):
    """这里 reward_model 实际上是 ground_truth"""
    ctx = _ensure_context()
    pad_token_id = ctx.pad_token_id
    rewards: list[float] | list[list[float]] = []
    format_rewards = format_reward(
        completions,
        completion_token_ids=completion_token_ids,
        prompts=prompts,
        data_source=data_source,
    )
    if completion_token_ids is None:
        raise RuntimeError("completion_token_ids must be provided for token matching.")

    for i, (tokens, rm, fr) in enumerate(
        zip(completion_token_ids, reward_model, format_rewards, strict=False)
    ):
        gt_tokens = _extract_gt_tokens(rm)
        norm_gt = _strip_padding_tokens(gt_tokens, pad_token_id)
        norm_tokens = _strip_padding_tokens(tokens, pad_token_id)

        if use_prm:
            if fr <= 0:
                length = max(len(norm_tokens), 1)
                rewards.append([float(fr)] * length)
                continue

            compare_len = _prm_compare_length(norm_tokens, norm_gt)
            length = max(len(norm_tokens), compare_len, 1)
            token_rewards = [0.0] * length
            for idx in range(compare_len):
                token_rewards[idx] = (
                    1.0 if norm_tokens[idx] == norm_gt[idx] else 0.0
            )
            rewards.append(token_rewards)
        else:
            if norm_tokens == norm_gt and fr > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    return rewards


def format_reward(
    completions: Iterable[list[dict[str, str]]],
    completion_token_ids: Iterable[list[int]] | None = None,
    prompts=None,
    data_source: Iterable[str] | None = None,
    **unused,
):
    r"""
    如果是 seqrec, 要符合 <a_*><b_*><c*_><d*_><|im_end|> 的格式,总共只能有 5 个 token,不能有 \n.
    否则直接给 -1.0 分.
    hints: 如果这里错误的给 0.0,不能修复 format, 因为内容错误在 rule_reward 里也会拿到 0.0 分.
    """
    rewards: list[float] = []
    ds_iter = data_source if data_source is not None else repeat(None)
    for completion, ds in zip(completions, ds_iter, strict=False):
        if ds in ["seqrec", "fusionseqrec"]:
            content = completion[0]["content"]
            # import pdb; pdb.set_trace()
            if _is_valid_seqrec_content(content):
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        else:
            rewards.append(1.0)
    return rewards


def _is_valid_seqrec_content(content: str) -> bool:
    if "\n" in content:
        return False
    pattern = _SEQREC_PATTERN or _build_seqrec_pattern(None)
    return bool(pattern.match(content.strip()))


def _ensure_context() -> _RewardContext:
    if _REWARD_CONTEXT is None:
        msg = "Call initialize_reward_functions() before invoking reward functions."
        raise RuntimeError(msg)
    return _REWARD_CONTEXT


if __name__ == "__main__":
    # ===== 基础初始化：带上 pad_token_id 方便正则忽略结尾填充 =====
    initialize_reward_functions(
        num_generations=4, pad_token_id=0, pad_token="<|endoftext|>"
    )
    """快速自检：覆盖格式/匹配/PRM 的典型场景，便于本地 sanity check。"""
    ctx = _ensure_context()
    # ===== 场景 1：格式 + 整体匹配/不匹配 =====
    valid_samples = [
        "<a_54><b_41><c_94><d_175><|im_end|>",
        "<a_93><b_150><c_157><d_155><|im_end|>",
        "<a_38><b_30><c_107><d_30><|im_end|>",
        "<a_90><b_147><c_205><d_16><|im_end|>",
    ]
    invalid_samples = [
        "<a_54><b_41><c_94><d_175><|im_start|>",
        "<a_54><b_41><c_94><d_175>",
        "<a_54><b_41><c_94><d_175>\n<|im_start|>",
        "<a_54><b_41><c_94><d_175><|im_end|>\n",
    ]

    completions = []
    reward_model = []
    perfect_match = valid_samples[0].split("<|im_end|>")[0]
    mismatch_gt = "<a_0><b_0><c_0><d_0><|im_end|>"

    for i in range(ctx.num_generations):
        sample = valid_samples[i % len(valid_samples)]
        completions.append([{"content": sample}])
        reward_model.append({"ground_truth": perfect_match if i == 0 else mismatch_gt})

    for i in range(ctx.num_generations):
        sample = invalid_samples[i % len(invalid_samples)]
        completions.append([{"content": sample}])
        reward_model.append({"ground_truth": mismatch_gt})

    data_source = ["seqrec"] * (2 * ctx.num_generations)

    format_scores = format_reward(completions, data_source=data_source)
    expected_format = [1.0] * ctx.num_generations + [-1.0] * ctx.num_generations

    rule_scores = rule_reward(
        reward_model,
        completions,
        data_source=data_source,
    )
    expected_rule = [1.0] + [0.0] * (2 * ctx.num_generations - 1)

    ndcg_scores = ndcg_rule_reward(
        reward_model,
        completions,
        data_source=data_source,
    )
    expected_ndcg = (
        [0.0]
        + [ctx.ndcg_rewards[i] for i in range(1, ctx.num_generations)]
        + [0.0] * ctx.num_generations
    )

    print("场景1 Completions:", completions)
    print("场景1 Ground truths:", reward_model)
    print("-" * 40)
    print("Format:", format_scores)
    print("Rule:", rule_scores)
    print("NDCG:", ndcg_scores)

    mismatches = []
    if format_scores != expected_format:
        mismatches.append(
            f"format_reward mismatch: expected {expected_format}, got {format_scores}"
        )
    if rule_scores != expected_rule:
        mismatches.append(
            f"rule_reward mismatch: expected {expected_rule}, got {rule_scores}"
        )
    if ndcg_scores != expected_ndcg:
        mismatches.append(
            f"ndcg_rule_reward mismatch: expected {expected_ndcg}, got {ndcg_scores}"
        )

    # ===== 场景 2：PRM token 级别测试（含 pad、格式错误） =====
    prm_gt_tokens = [10, 20, 30, 40, 0]  # 最后一个 0 视为 pad
    prm_reward_model = [
        {"ground_truth": {"text": "gt", "token": prm_gt_tokens}}
        for _ in range(ctx.num_generations)
    ]
    prm_completions = [
        [{"content": "<a_1><b_1><c_1><d_1><|im_end|>"}],  # 全部匹配
        [{"content": "<a_1><b_2><c_1><d_1><|im_end|>"}],  # 第2个 token 错
        [{"content": "<a_9><b_1><c_1><d_1><|im_end|>"}],  # 第1个 token 错
        [{"content": "<a_1><b_1><c_1><d_1><|im_start|>"}],  # 格式非法
    ]
    prm_completion_tokens = [
        [10, 20, 30, 40, 0],
        [10, 99, 30, 40, 0],
        [99, 20, 30, 40, 0],
        [10, 20, 30, 40, 0],
    ]
    prm_data_source = ["seqrec"] * ctx.num_generations

    prm_format = format_reward(prm_completions, data_source=prm_data_source)
    prm_rule = rule_reward(
        prm_reward_model,
        prm_completions,
        completion_token_ids=prm_completion_tokens,
        data_source=prm_data_source,
        use_prm=True,
    )
    prm_ndcg = ndcg_rule_reward(
        prm_reward_model,
        prm_completions,
        completion_token_ids=prm_completion_tokens,
        data_source=prm_data_source,
        use_prm=True,
    )

    expected_prm_format = [1.0, 1.0, 1.0, -1.0]
    expected_prm_rule = [
        [1.0, 1.0, 1.0, 1.0],  # 完全匹配
        [1.0, 0.0, 1.0, 1.0],  # 第二位错误
        [0.0, 1.0, 1.0, 1.0],  # 第一位错误
        [-1.0, -1.0, -1.0, -1.0],  # 格式错误 -> 直接格式分
    ]
    expected_prm_ndcg = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, ctx.ndcg_rewards[1], 0.0, 0.0],
        [ctx.ndcg_rewards[2], 0.0, 0.0, 0.0],
        [-1.0, -1.0, -1.0, -1.0],
    ]

    if prm_format != expected_prm_format:
        mismatches.append(
            f"PRM format mismatch: expected {expected_prm_format}, got {prm_format}"
        )
    if prm_rule != expected_prm_rule:
        mismatches.append(
            f"PRM rule mismatch: expected {expected_prm_rule}, got {prm_rule}"
        )
    if prm_ndcg != expected_prm_ndcg:
        mismatches.append(
            f"PRM ndcg mismatch: expected {expected_prm_ndcg}, got {prm_ndcg}"
        )

    if mismatches:
        msg = "\n".join(mismatches)
        raise AssertionError(f"Reward sanity check failed:\n{msg}")

    print("Reward sanity check passed with seqrec + PRM samples.")
