# multimodal_test.py
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ..collator import TestCollator
from ..config import parse_args
from ..evaluate import get_metrics_results, get_topk_results
from ..prompt import all_prompt
from ..type import Args
from ..utils import (
    load_model_for_inference,
    load_test_dataset,
    set_seed,
)


def setup_test(
    model_type: str,
    model_path: str,
    ckpt_path: str,
    use_lora: bool,
    gpu_id: int,
    seed: int,
    args: Args,
) -> tuple[
    Any,
    AutoTokenizer,
    torch.device,
]:
    """
    初始化单个模型的测试环境.

    """
    set_seed(seed)

    device = torch.device("cuda", gpu_id)

    model, tokenizer_or_processor = load_model_for_inference(
        model_type=model_type,
        model_path=model_path,
        ckpt_path=ckpt_path,
        use_lora=use_lora,
    )
    if args.test_args.models[0].model_type == "qwen_vl":
        tokenizer = tokenizer_or_processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def get_prompt_ids(args: Args) -> list[int]:
    """
    获取测试的prompt ID列表.

    all_prompt 有 3 个 task:
    seqrec, itemsearch, fusionseqrec

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
    tokenizer: AutoTokenizer,
) -> tuple[DataLoader, list, Callable]:
    """
    准备测试数据.
    """
    test_data = load_test_dataset(args)
    collator = TestCollator(args.dataset_args, tokenizer)
    # else:
    # collator = UnifiedTestCollator(args.dataset_args, tokenizer_or_processor)
    all_items = test_data.get_all_items()
    prefix_allowed_tokens = (
        test_data.get_prefix_allowed_tokens_fn(tokenizer)
        if args.test_args.use_constrained_generation
        else None
    )
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
    tokenizer: AutoTokenizer,
    num_beams: int,
):
    """为有命中的样本打印详细信息。"""
    print("\n")
    # 解码输入
    input_text = tokenizer.decode(
        inputs["input_ids"][index], skip_special_tokens=True
    )
    # print(f"Input: {input_text}")
    # print("-" * 50)
    # print(f"Target: {targets[index]}")
    # print("-" * 50)

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
    tokenizer: AutoTokenizer,
    prefix_allowed_tokens: Callable,
    all_items: list,
    metrics: list[str],
    args: Args,
) -> dict[str, float]:
    """
    评估单个prompt.
    """
    print(f"\n开始测试 Prompt {prompt_id}...")
    test_loader.dataset.set_prompt(prompt_id)
    metrics_results = {}
    total = 0
    test_args = args.test_args
    global_args = args.global_args

    for step, batch in enumerate(tqdm(test_loader, desc=f"Prompt {prompt_id}")):
        inputs, targets = batch
        total += len(targets)
        if global_args.debug and step == 0:
            # 打印输入信息
            print("\n[1. 输入信息]")
            print(f"  - 批次大小 (Batch Size): {len(targets)}")
            # print(f"  - 真实目标 (Targets): {targets}")
            # print(f"  - 输入: {inputs}")
            first_sample_text = tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=False
            )
            print(f"  - 第一个样本的文本输入 (解码后):\n'{first_sample_text}'")

        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        if global_args.debug:
            print(
                "test_args.use_constrained_generation:",
                test_args.use_constrained_generation,
            )
        output: dict[str, torch.Tensor] = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=test_args.max_new_tokens,
            num_beams=test_args.num_beams,
            num_return_sequences=test_args.num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
        )

        output_ids = output["sequences"]
        scores = output["sequences_scores"]
        output_texts = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        if global_args.debug and step == 0:
            print("  - 模型原始输出 (解码后):")
            for i in range(len(targets)):
                print(f"    - 样本 {i + 1} (目标: {targets[i]}):")
                sample_predictions = output_texts[
                    i * test_args.num_beams : (i + 1) * test_args.num_beams
                ]
                sample_scores = scores[
                    i * test_args.num_beams : (i + 1) * test_args.num_beams
                ]
                for j, (pred, score) in enumerate(
                    zip(sample_predictions, sample_scores, strict=False)
                ):
                    # 清理预测文本以提高可读性
                    clean_pred = pred.split("Response:")[-1].strip()
                    print(
                        f"      - Beam {j + 1}: '{clean_pred}' (Score: {score:.4f})"
                    )
        # 获得 topk 结果
        topk_res = get_topk_results(
            output_texts,
            scores,
            targets,
            test_args.num_beams,
            all_items=all_items if test_args.filter_items else None,
        )
        # 获得 metrics 结果
        batch_metrics_res = get_metrics_results(topk_res, metrics)

        for m, res in batch_metrics_res.items():
            if m not in metrics_results:
                metrics_results[m] = res
            else:
                metrics_results[m] += res

        if (step + 1) % test_args.print_freq == 0:
            current_metrics = {k: v / total for k, v in metrics_results.items()}
            print(f"Step {step + 1}, 当前指标: {current_metrics}")

        if global_args.debug:
            print("\n[DEBUG MODE] 只处理一个批次。")
            # 可以在这里加入更详细的debug信息，比如打印topk_res
            print_hit_details(
                0,
                topk_res[0],
                inputs,
                targets,
                output_texts,
                scores,
                tokenizer,
                test_args.num_beams,
            )
            break

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
    model_config: Any,
    tokenizer: AutoTokenizer,
) -> None:
    """
    汇总并保存结果.
    """
    mean_results, min_results, max_results = {}, {}, {}
    for m in metrics:
        all_res = [res[m] for res in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("=" * 60)
    print(f"模型 [{model_config.name}] 的平均结果: ", mean_results)
    print("=" * 60)

    # 动态构建结果文件路径
    output_root = Path(args.global_args.output_dir)
    model_result_dir = (
        output_root / args.dataset_args.dataset / model_config.name
    )
    model_result_dir.mkdir(parents=True, exist_ok=True)
    results_file = model_result_dir / "test_results.json"

    save_data = {
        "model_name": model_config.name,
        "test_task": args.test_args.test_task,
        "test_prompt_ids": args.test_args.test_prompt_ids,
        "mean_results": mean_results,
        "min_results": min_results,
        "max_results": max_results,
        "all_prompt_results": all_prompt_results,
        "model_info": {
            "model_type": model_config.model_type,
            "base_model": model_config.path,
            "ckpt_path": model_config.ckpt_path,
            "lora": model_config.lora,
            "vocab_size": len(tokenizer),
        },
        "test_config": {
            "num_beams": args.test_args.num_beams,
            "max_new_tokens": args.test_args.max_new_tokens,
            "test_batch_size": args.test_args.test_batch_size,
            "use_constrained_generation": args.test_args.use_constrained_generation,
        },
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"结果已保存到: {results_file}")


def main() -> None:
    """
    执行完整的测试流程.
    """
    args = parse_args()
    prompt_ids = get_prompt_ids(args)
    metrics: list[str] = args.test_args.metrics.split(",")

    for model_config in args.test_args.models:
        if not model_config.enabled:
            continue

        print(f"\n{'=' * 25} 开始评测模型: {model_config.name} {'=' * 25}\n")

        # 1. 覆盖数据集相关的模型特定参数
        if hasattr(model_config, "index_file") and model_config.index_file:
            args.dataset_args.index_file = model_config.index_file

        # 2. 初始化
        model, tokenizer, device = setup_test(
            model_type=model_config.model_type,
            model_path=model_config.path,
            ckpt_path=model_config.ckpt_path,
            use_lora=model_config.lora,
            gpu_id=args.test_args.gpu_id,
            seed=args.global_args.seed,
            args=args,
        )
        # 3. 准备数据
        test_loader, all_items, prefix_allowed_tokens = prepare_test_data(
            args, tokenizer
        )

        # 4. 执行评测
        model.eval()
        all_prompt_results: list[dict[str, float]] = []
        with torch.no_grad():
            for prompt_id in prompt_ids:
                metrics_results = evaluate_prompt(
                    prompt_id,
                    model,
                    test_loader,
                    device,
                    tokenizer,
                    prefix_allowed_tokens,
                    all_items,
                    metrics,
                    args,
                )
                all_prompt_results.append(metrics_results)

        # 5. 汇总并保存结果
        if all_prompt_results:
            summarize_and_save_results(
                all_prompt_results,
                metrics,
                args,
                model_config,
                tokenizer,
            )
        else:
            print(f"模型 {model_config.name} 没有可供汇总的结果。")

        print(f"\n{'=' * 25} 模型: {model_config.name} 评测完成 {'=' * 25}\n")


if __name__ == "__main__":
    main()
