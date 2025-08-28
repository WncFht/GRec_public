import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..collator import UnifiedTestCollator
from ..data import SeqRecDataset
from ..evaluate import get_metrics_results, get_topk_results
from ..parser import parse_dataset_args, parse_global_args, parse_test_args
from ..prompt import all_prompt
from ..utils import load_model_for_inference, set_seed


def test(args: argparse.Namespace):
    """使用LoRA模型进行完整的评估测试"""
    set_seed(args.seed)
    print("=" * 80)
    print("LoRA模型评估测试")
    print("=" * 80)
    print(vars(args))

    device = torch.device("cuda", args.gpu_id)

    # 使用load_model_for_inference加载模型
    print("\n加载模型...")
    model, processor = load_model_for_inference(
        model_type=args.model_type,
        ckpt_path=args.ckpt_path,
        use_lora=args.lora,
        model_path=args.base_model if args.lora else None,
    )

    # 确保模型在正确的设备上
    if not hasattr(model, "device"):
        model.to(device)

    # 设置tokenizer
    tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 确定要测试的prompt
    if args.test_prompt_ids == "all":
        prompt_ids = range(len(all_prompt["seqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    # 准备数据集和数据加载器
    test_data = SeqRecDataset(args, mode="test")
    collator = UnifiedTestCollator(args, processor_or_tokenizer=processor)
    all_items = test_data.get_all_items()

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\n数据集大小: {len(test_data)}")
    print(f"测试批次大小: {args.test_batch_size}")
    print(f"测试prompt IDs: {prompt_ids}")

    model.eval()

    # 解析评估指标
    metrics = args.metrics.split(",")
    all_prompt_results = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            print(f"\n评估Prompt {prompt_id}...")
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(
                tqdm(test_loader, desc=f"Prompt {prompt_id}")
            ):
                inputs = batch[0]
                targets = batch[1]
                total += len(targets)

                # 将输入移到设备
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 生成输出
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=4,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )

                output_ids = output["sequences"]
                scores = output["sequences_scores"].cpu().numpy()

                # 批量解码输出
                output_texts = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                # 提取响应部分
                all_outputs = []
                for i in range(0, len(output_texts), args.num_beams):
                    beam_outputs = []
                    for j in range(args.num_beams):
                        text = output_texts[i + j]
                        response = text.split("Response:")[-1].strip()
                        beam_outputs.append(response)
                    all_outputs.append(beam_outputs)

                # 获取top-k结果
                topk_results = get_topk_results(
                    all_outputs,
                    targets,
                    scores.reshape(-1, args.num_beams),
                    all_items,
                    filter_items=args.filter_items,
                )

                # 更新评估指标
                batch_metrics = get_metrics_results(topk_results, metrics)
                for metric in metrics:
                    if metric not in metrics_results:
                        metrics_results[metric] = 0
                    metrics_results[metric] += batch_metrics[metric]

                # 打印进度
                if (step + 1) % args.print_freq == 0:
                    partial_results = {
                        m: metrics_results[m]
                        / ((step + 1) * args.test_batch_size)
                        for m in metrics
                    }
                    print(f"Step {step + 1}: {partial_results}")

            # 计算最终指标
            for metric in metrics:
                metrics_results[metric] = metrics_results[metric] / total

            print(f"Prompt {prompt_id} 结果: {metrics_results}")

            prompt_result = {
                "prompt_id": prompt_id,
                "prompt": all_prompt["seqrec"][prompt_id]["user"],
                "metrics": metrics_results,
                "total_samples": total,
            }
            all_prompt_results.append(prompt_result)

    # 计算平均结果
    print("\n" + "=" * 80)
    print("最终结果:")
    print("=" * 80)

    avg_metrics = {}
    for metric in metrics:
        avg_metrics[metric] = sum(
            r["metrics"][metric] for r in all_prompt_results
        ) / len(all_prompt_results)

    print("\n各Prompt结果:")
    for result in all_prompt_results:
        print(f"  Prompt {result['prompt_id']}: {result['metrics']}")

    print(f"\n平均结果: {avg_metrics}")

    # 保存结果
    results = {
        "model_type": args.model_type,
        "checkpoint": args.ckpt_path,
        "is_lora": args.lora,
        "base_model": args.base_model if args.lora else None,
        "dataset": args.dataset,
        "prompt_results": all_prompt_results,
        "average_metrics": avg_metrics,
        "test_config": {
            "num_beams": args.num_beams,
            "batch_size": args.test_batch_size,
            "filter_items": args.filter_items,
        },
    }

    # 确保结果目录存在
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    # 为LoRA结果添加特殊标记
    if args.lora:
        base_name, ext = os.path.splitext(args.results_file)
        results_file = f"{base_name}_lora{ext}"
    else:
        results_file = args.results_file

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {results_file}")

    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA模型评估测试")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test(args)
