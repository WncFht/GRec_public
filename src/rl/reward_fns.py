from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import repeat
from typing import Any

from ..utils import clean_text


@dataclass
class _RewardContext:
    num_generations: int
    ndcg_rewards: list[float]


_REWARD_CONTEXT: _RewardContext | None = None


def initialize_reward_functions(num_generations: int) -> bool:
    """Prepare reward helpers and optionally run a quick sanity check."""
    global _REWARD_CONTEXT
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    ndcg_rewards = [-elm / sum(ndcg_rewards) for elm in ndcg_rewards]
    _REWARD_CONTEXT = _RewardContext(num_generations, ndcg_rewards)

    return False


def ndcg_rule_reward(
    reward_model: Iterable[dict[str, Any]],
    completions: Iterable[list[dict[str, str]]],
    prompts=None,
    data_source: str | None = None,
    **unused,
):
    ctx = _ensure_context()
    repeat = ctx.num_generations
    rewards: list[float] = []
    flag = False
    lis: list[float] = []

    format_rewards = format_reward(
        completions, prompts, data_source=data_source
    )
    for i, (completion, rm, fr) in enumerate(
        zip(completions, reward_model, format_rewards, strict=False)
    ):
        if (
            clean_text(completion[0]["content"])
            == clean_text(rm["ground_truth"])
            and fr != 0
        ):
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
    prompts=None,
    data_source: str | None = None,
    **unused,
):
    """这里 reward_model 实际上是 ground_truth"""
    rewards: list[float] = []
    format_rewards = format_reward(
        completions, prompts, data_source=data_source
    )
    for i, (completion, rm, fr) in enumerate(
        zip(completions, reward_model, format_rewards, strict=False)
    ):
        if (
            clean_text(completion[0]["content"])
            == clean_text(rm["ground_truth"])
            and fr != 0
        ):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


_SEQREC_PATTERN = re.compile(
    r"^<a_[^<>\n]+><b_[^<>\n]+><c_[^<>\n]+><d_[^<>\n]+><\|im_end\|>$"
)


def format_reward(
    completions: Iterable[list[dict[str, str]]],
    prompts=None,
    data_source: Iterable[str] | None = None,
    **unused,
):
    r"""如果是 seqrec, 要符合 <a_*><b_*><c*_><d*_><|im_end|> 的格式,总共只能有 5 个 token,不能有 \n"""
    rewards: list[float] = []
    ds_iter = data_source if data_source is not None else repeat(None)
    for completion, ds in zip(completions, ds_iter, strict=False):
        if ds == "seqrec":
            content = completion[0]["content"]
            # import pdb; pdb.set_trace()
            if _is_valid_seqrec_content(content):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def _is_valid_seqrec_content(content: str) -> bool:
    if "\n" in content:
        return False
    return bool(_SEQREC_PATTERN.match(content.strip()))


def _ensure_context() -> _RewardContext:
    if _REWARD_CONTEXT is None:
        msg = "Call initialize_reward_functions() before invoking reward functions."
        raise RuntimeError(msg)
    return _REWARD_CONTEXT


if __name__ == "__main__":
    initialize_reward_functions(num_generations=4)
    """Quick helper that runs the reward funcs on known-valid/invalid seqrec outputs."""
    ctx = _ensure_context()
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
        reward_model.append(
            {"ground_truth": perfect_match if i == 0 else mismatch_gt}
        )

    for i in range(ctx.num_generations):
        sample = invalid_samples[i % len(invalid_samples)]
        completions.append([{"content": sample}])
        reward_model.append({"ground_truth": mismatch_gt})

    data_source = ["seqrec"] * (2 * ctx.num_generations)

    format_scores = format_reward(completions, data_source=data_source)
    expected_format = [1.0] * ctx.num_generations + [0.0] * ctx.num_generations

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

    print("Completions tested:", completions)
    print("Ground truths tested:", reward_model)
    print("-" * 40)
    print("Format scores:", format_scores)
    print("Expected format scores:", expected_format)
    print("-" * 40)
    print("Rule scores:", rule_scores)
    print("Expected rule scores:", expected_rule)
    print("-" * 40)
    print("NDCG scores:", ndcg_scores)
    print("Expected NDCG scores:", expected_ndcg)

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

    if mismatches:
        msg = "\n".join(mismatches)
        raise AssertionError(f"Reward sanity check failed:\n{msg}")

    print("Reward sanity check passed with provided seqrec samples.")
