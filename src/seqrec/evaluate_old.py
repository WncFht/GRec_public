# multimodal_test.py
import argparse
import json
import os
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from collator import UnifiedTestCollator
from config import parse_args
from evaluate import get_metrics_results, get_topk_results
from prompt import all_prompt
from type import Args
from utils import (
    get_tokenizer,
    load_model_for_inference,
    load_test_dataset,
    set_seed,
)


def setup_test(
    args: Args,
) -> tuple[
    Any,
    AutoProcessor | AutoTokenizer,
    torch.device,
]:
    """
    初始化测试环境.

    Args:
    ----
        args (Args): 包含种子、GPU ID等设置的参数.

    Returns:
    -------
        tuple[Any, Union[AutoProcessor, AutoTokenizer], torch.device]:
            初始化后的模型、处理器/tokenizer和设备.

    """
    set_seed(args.global_args.seed)
    print(vars(args))

    device = torch.device("cuda", args.test_args.gpu_id)

    model, tokenizer_or_processor = load_model_for_inference(args)
    tokenizer = get_tokenizer(tokenizer_or_processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer_or_processor, device


def get_prompt_ids(args: Args) -> list[int]:
    """
    获取测试的prompt ID列表.

    all_prompt 有 3 个 task:
    seqrec, itemsearch, fusionseqrec

    Args:
    ----
        args (Args): 包含 test_prompt_ids 和 test_task 的参数.

    Returns:
    -------
        list[int]: 测试的prompt ID列表.

    """
    test_args = args.test_args
    if test_args.test_prompt_ids == "all":
        task = test_args.test_task.lower()
        if task in all_prompt:
            return range(len(all_prompt[task]))
        raise ValueError(f"不支持的测试任务: {task}")
    # 挑具体的那个 test_task 的 test_prompt_ids 的 prompt
    return [int(_) for _ in test_args.test_prompt_ids.split(",")]


def prepare_test_data(
    args: Args,
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
) -> tuple[DataLoader, list, Callable]:
    """
    准备测试数据.

    Args:
    ----
        args (Args): 包含数据集配置的参数.
        tokenizer_or_processor (Union[AutoProcessor, AutoTokenizer]): 用于数据处理的处理器或tokenizer.

    Returns:
    -------
        tuple[DataLoader, list, Callable]: 测试数据加载器、所有物品列表和前缀允许的tokens函数.

    """
    test_data = load_test_dataset(args)
    collator = UnifiedTestCollator(args, tokenizer_or_processor)
    all_items = test_data.get_all_items()
    tokenizer = get_tokenizer(tokenizer_or_processor)
    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print("测试数据数量:", len(test_data))
    return test_loader, all_items, prefix_allowed_tokens


def print_hit_details(
    index: int,
    topk_hit_info: list[int],
    inputs: dict[str, torch.Tensor],
    targets: list[str],
    output_texts: list[str],
    scores: torch.Tensor,
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
    num_beams: int,
):
    """为有命中的样本打印详细信息。"""
    print("\n")
    tokenizer = get_tokenizer(tokenizer_or_processor)
    # 解码输入
    input_text = tokenizer.decode(
        inputs["input_ids"][index], skip_special_tokens=True
    )
    print(f"Input: {input_text}")
    print("-" * 50)
    print(f"Target: {targets[index]}")
    print("-" * 50)

    # 提取并排序当前样本的预测结果
    start_idx = index * num_beams
    end_idx = (index + 1) * num_beams
    sample_preds = output_texts[start_idx:end_idx]
    sample_scores = scores[start_idx:end_idx].cpu().numpy()

    sorted_responses = sorted(
        zip(sample_preds, sample_scores, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )

    # 打印得分最高的结果
    best_response_text = sorted_responses[0][0]
    print(f"Best Response: \n{best_response_text}")
    print("-" * 50)
    print(f"Top-k Output: {topk_hit_info}")
    print("-" * 50)


def evaluate_prompt(
    prompt_id: int,
    model: Any,
    test_loader: DataLoader,
    device: torch.device,
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
    args: Args,
    all_items: list,
    metrics: list[str],
) -> dict[str, float]:
    """
    评估单个prompt.

    Args:
    ----
        prompt_id (int): prompt id.
        model (Any): 模型.
        test_loader (DataLoader): 测试数据.
        device (torch.device): 设备.
        tokenizer_or_processor (Union[AutoProcessor, AutoTokenizer]): 处理器或tokenizer.
        args (Args): 参数.
        all_items (list): 所有物品列表.
        metrics (list[str]): 评价指标列表.

    Returns:
    -------
        dict[str, float]: 包含各项指标结果的字典.

    """
    print(f"\n开始测试 Prompt {prompt_id}...")
    test_loader.dataset.set_prompt(prompt_id)
    metrics_results = {}
    total = 0
    test_args = args.test_args

    for step, batch in enumerate(tqdm(test_loader, desc=f"Prompt {prompt_id}")):
        # for step, batch in enumerate(test_loader):
        inputs, targets = batch
        total += len(targets)

        # 将 inputs 中的所有张量移动到 device 上
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        tokenizer = get_tokenizer(tokenizer_or_processor)
        prefix_allowed_tokens_fn = (
            test_loader.dataset.get_prefix_allowed_tokens_fn(tokenizer)
            if test_args.use_constrained_generation
            else None
        )

        output: dict[str, torch.Tensor] = model.generate(
            **inputs,
            max_new_tokens=test_args.max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=test_args.num_beams,
            num_return_sequences=test_args.num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
        )

        output_ids = output["sequences"]
        scores = output["sequences_scores"]
        output_texts = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        # 获得 topk 结果
        topk_res = get_topk_results(
            output_texts,
            scores,
            targets,
            test_args.num_beams,
            all_items=all_items
            if "filter_items" in test_args and test_args.filter_items
            else None,
            model_type=args.global_args.model_type,
        )
        # 获得 metrics 结果
        batch_metrics_res = get_metrics_results(topk_res, metrics)

        for m, res in batch_metrics_res.items():
            metrics_results[m] = metrics_results.get(m, 0) + res

        if (step + 1) % test_args.print_freq == 0:
            current_metrics = {k: v / total for k, v in metrics_results.items()}
            print(f"Step {step + 1}, 当前指标: {current_metrics}")

    final_metrics = {k: v / total for k, v in metrics_results.items()}
    print("=" * 60)
    print(f"Prompt {prompt_id} 结果: {final_metrics}")
    print("=" * 60)
    print()
    return final_metrics


def summarize_and_save_results(
    all_prompt_results: list[dict[str, float]],
    metrics: list[str],
    args: Args,
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
) -> None:
    """
    汇总并保存结果.

    Args:
    ----
        all_prompt_results (list[dict[str, float]]): 所有 prompt 的评测结果.
        metrics (list[str]): 评价指标列表.
        args (Args): 参数.
        tokenizer_or_processor (Union[AutoProcessor, AutoTokenizer]): 处理器或tokenizer.

    """
    mean_results, min_results, max_results = {}, {}, {}
    for m in metrics:
        all_res = [res[m] for res in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("=" * 60)
    print("平均结果: ", mean_results)
    print("最小结果: ", min_results)
    print("最大结果: ", max_results)
    print("=" * 60)

    test_args = args.test_args
    global_args = args.global_args

    save_data = {
        "test_task": test_args.test_task,
        "test_prompt_ids": test_args.test_prompt_ids,
        "mean_results": mean_results,
        "min_results": min_results,
        "max_results": max_results,
        "all_prompt_results": all_prompt_results,
        "model_info": {
            "model_type": global_args.model_type,
            "base_model": global_args.base_model,
            "ckpt_path": test_args.ckpt_path,
            "lora": test_args.lora,
            "vocab_size": len(get_tokenizer(tokenizer_or_processor)),
        },
        "test_config": {
            "num_beams": test_args.num_beams,
            "max_new_tokens": test_args.max_new_tokens,
            "test_batch_size": test_args.test_batch_size,
            "use_constrained_generation": test_args.use_constrained_generation,
        },
    }

    os.makedirs(os.path.dirname(test_args.results_file), exist_ok=True)
    with open(test_args.results_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"结果已保存到: {test_args.results_file}")


def test(args: Args) -> None:
    """
    执行完整的测试流程.

    Args:
    ----
        args (Args): 包含所有测试配置的参数.

    """
    # 1. 初始化
    model, tokenizer_or_processor, device = setup_test(args)

    # 2. 获取 prompts
    prompt_ids = get_prompt_ids(args)

    # 3. 准备数据
    test_loader, all_items, _ = prepare_test_data(args, tokenizer_or_processor)

    # 4. 执行评测
    model.eval()
    # metrics 有 hit@1, hit@5, hit@10, ndcg@5, ndcg@10 等
    metrics: list[str] = args.test_args.metrics.split(",")
    all_prompt_results: list[dict[str, float]] = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            metrics_results = evaluate_prompt(
                prompt_id,
                model,
                test_loader,
                device,
                tokenizer_or_processor,
                args,
                all_items,
                metrics,
            )
            all_prompt_results.append(metrics_results)

    # 5. 汇总并保存结果
    if all_prompt_results:
        summarize_and_save_results(
            all_prompt_results, metrics, args, tokenizer_or_processor
        )
    else:
        print("没有可供汇总的结果。")


if __name__ == "__main__":
    # 使用新的配置系统
    args = parse_args()
    # 兼容旧的命令行参数（如果存在）
    # 注意：这只是为了平滑过渡，理想情况下所有配置都应在YML中
    parser = argparse.ArgumentParser(
        description="MultiModal Recommendation Model Test"
    )
    parser.add_argument(
        "--use_constrained_generation",
        action="store_true",
        help="Whether to use constrained generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--print_freq", type=int, default=4, help="Print frequency"
    )

    # 只解析额外的参数
    extra_args, _ = parser.parse_known_args()

    # 将额外参数更新到从YML加载的args对象中
    # 这里我们假设这些是test_args的一部分
    if extra_args.use_constrained_generation:
        args.test_args.use_constrained_generation = True
    if extra_args.max_new_tokens is not None:
        args.test_args.max_new_tokens = extra_args.max_new_tokens
    if extra_args.print_freq is not None:
        args.test_args.print_freq = extra_args.print_freq

    test(args)
