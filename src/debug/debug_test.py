# src/debug_test.py
import argparse
import json

import torch
from evaluate import get_metrics_results, get_topk_results
from multimodal_test import (
    get_prompt_ids,
    get_tokenizer,
    prepare_test_data,
    setup_test,
)
from utils import parse_dataset_args, parse_global_args, parse_test_args


def debug_test(args: argparse.Namespace) -> None:
    """
    对模型进行单步调试，检查单个批次的输入、输出和评估指标计算过程。

    Args:
    ----
        args (argparse.Namespace): 包含所有配置的参数。

    """
    # 1. 初始化模型、处理器和设备
    model, tokenizer_or_processor, device = setup_test(args)
    model.eval()
    tokenizer = get_tokenizer(tokenizer_or_processor)

    # 2. 选择一个 prompt进行测试
    prompt_ids = get_prompt_ids(args)
    if not prompt_ids:
        print(
            "错误: 未根据 `test_task` 和 `test_prompt_ids` 找到可用的 prompt。"
        )
        return

    # 为了调试，我们只使用找到的第一个 prompt
    prompt_id_to_test = prompt_ids[0]
    print(f"--- 将使用 Prompt ID: {prompt_id_to_test} 进行调试 ---")

    # 3. 准备数据加载器
    test_loader, all_items, _ = prepare_test_data(args, tokenizer_or_processor)

    # 为数据集设置指定的 prompt
    test_loader.dataset.set_prompt(prompt_id_to_test)

    # 4. 从数据加载器中获取一个样本批次
    try:
        sample_batch = next(iter(test_loader))
    except StopIteration:
        print("错误: 测试数据加载器为空，无法获取样本。")
        return

    print(f"\n--- 开始调试单个批次 (Batch Size={args.test_batch_size}) ---")

    with torch.no_grad():
        inputs, targets = sample_batch

        # --- 打印输入信息 ---
        print("\n[1. 输入信息]")
        print(f"  - 批次大小 (Batch Size): {len(targets)}")
        print(f"  - 真实目标 (Targets): {targets}")
        print(f"  - 输入: {inputs}")
        # 解码并打印批次中第一个样本的文本输入，以了解其构成
        first_sample_text = tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        print(f"  - 第一个样本的文本输入 (解码后):\n'{first_sample_text}'")
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            print(f"  - 图像数据维度: {inputs['pixel_values'].shape}")
        else:
            print("  - 无图像输入")

        # 将所有张量移动到指定设备
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # --- 模型生成 ---
        print("\n[2. 模型生成]")
        prefix_allowed_tokens_fn = (
            test_loader.dataset.get_prefix_allowed_tokens_fn(tokenizer)
            if args.use_constrained_generation
            else None
        )
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        output_ids = output["sequences"]
        scores = output["sequences_scores"]

        # --- 打印原始输出 ---
        output_texts = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        print("  - 模型原始输出 (解码后):")
        for i in range(len(targets)):
            print(f"    - 样本 {i + 1} (目标: {targets[i]}):")
            sample_predictions = output_texts[
                i * args.num_beams : (i + 1) * args.num_beams
            ]
            sample_scores = scores[
                i * args.num_beams : (i + 1) * args.num_beams
            ]
            for j, (pred, score) in enumerate(
                zip(sample_predictions, sample_scores, strict=False)
            ):
                # 清理预测文本以提高可读性
                clean_pred = pred
                # if args.model_type == "qwen_vl":
                clean_pred = pred.split("Response:")[-1].strip()
                print(
                    f"      - Beam {j + 1}: '{clean_pred}' (Score: {score:.4f})"
                )

        # --- 评估计算 ---
        print("\n[3. 评估计算]")

        # a. 调用 get_topk_results
        topk_res = get_topk_results(
            output_texts,
            scores.cpu().tolist(),  # 将scores移回CPU
            targets,
            args.num_beams,
            all_items=all_items
            if "filter_items" in args and args.filter_items
            else None,
            model_type=args.model_type,
        )
        print("  - `get_topk_results` 的输出 (命中列表):")
        for i, res in enumerate(topk_res):
            print(f"    - 样本 {i + 1} (目标: {targets[i]}): {res}")

        # b. 调用 get_metrics_results
        metrics_to_calc = args.metrics.split(",")
        batch_metrics_res = get_metrics_results(topk_res, metrics_to_calc)
        print("\n  - `get_metrics_results` 的输出 (批次指标总和):")
        print(f"    {json.dumps(batch_metrics_res, indent=4)}")

        # c. 为每个样本单独计算指标以方便验证
        print("\n  - 单样本指标 (用于验证):")
        for i, single_topk_res in enumerate(topk_res):
            single_metric = get_metrics_results(
                [single_topk_res], metrics_to_calc
            )
            print(f"    - 样本 {i + 1} (目标: {targets[i]}): {single_metric}")

    print("\n--- 调试结束 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiModal Recommendation Model Debug Script"
    )
    # 复用 multimodal_test.py 中的参数解析器
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    # 添加或覆盖部分参数以方便调试
    parser.add_argument(
        "--use_constrained_generation",
        action="store_true",
        help="是否使用受限生成 (例如，在物品推荐任务中)",
    )
    # 为了能看到生成的文本，可以适当增加 token 数量
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="生成的最大新 token 数量",
    )

    args = parser.parse_args()
    debug_test(args)
