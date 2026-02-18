import argparse
import csv
import json
import os
import re

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.collator import (
    ChatTemplateTestCollator,
    TestCollator,
    UnifiedTestCollator,
)
from src.evaluate import get_metrics_results, get_topk_results
from src.parser import parse_dataset_args, parse_global_args, parse_test_args
from src.prompt import all_prompt
from src.utils import load_model_for_inference, load_test_dataset, set_seed


def _to_metric_name(metric_key: str) -> str:
    metric = metric_key.strip().lower()
    if metric.startswith("hit@"):
        return f"HR@{metric.split('@', 1)[1]}"
    if metric.startswith("ndcg@"):
        return f"NDCG@{metric.split('@', 1)[1]}"
    return metric_key


def _round4(value):
    try:
        return round(float(value), 4)
    except Exception:
        return value


def _extract_topk(metric_keys: list[str]) -> list[int]:
    topk = set()
    for key in metric_keys:
        match = re.match(r"^(?:HR|NDCG)@(\d+)$", key)
        if match:
            topk.add(int(match.group(1)))
    return sorted(topk)


def _save_metrics_files(results_file: str, mean_results: dict):
    output_dir = os.path.dirname(results_file) or "."
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    metrics_tsv_path = os.path.join(output_dir, "metrics.tsv")

    metrics_for_save = {_to_metric_name(k): _round4(v) for k, v in mean_results.items()}
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_for_save, f, indent=2, ensure_ascii=False)

    topk = _extract_topk(list(metrics_for_save.keys()))
    columns = [f"HR@{k}" for k in topk] + [f"NDCG@{k}" for k in topk]
    columns = [c for c in columns if c in metrics_for_save]
    if not columns:
        columns = sorted(metrics_for_save.keys())

    row_path = output_dir
    row_name = row_path.split("/", 1)[1] if "/" in row_path else row_path
    with open(metrics_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", *columns])
        writer.writerow([row_name, *[metrics_for_save.get(c, "") for c in columns]])

    print(f"Metrics saved to: {metrics_json_path}, {metrics_tsv_path}")


def test(args: argparse.Namespace):
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False))
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
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 确定要测试的prompt
    if args.test_prompt_ids == "all":
        prompt_ids = range(len(all_prompt["seqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    # 准备数据集和数据加载器
    test_data = load_test_dataset(args)

    print("Num test data:", len(test_data))

    if args.model_type == "llama":
        collator = TestCollator(args, tokenizer=processor)
    elif args.model_type in ["qwen"]:
        collator = ChatTemplateTestCollator(args, tokenizer=processor)
    else:
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

            for step, batch in enumerate(tqdm(test_loader, desc=f"Prompt {prompt_id}")):
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
                    temperature=1.0,
                )

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                # print(output)
                # print(scores)
                topk_res = get_topk_results(
                    output,
                    scores,
                    targets,
                    args.num_beams,
                    all_items=all_items if args.filter_items else None,
                )

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                # print(batch_metrics_res)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % 10 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            print("======================================================")
            print(args.ckpt_path)
            print("======================================================")
            print(f"Prompt {prompt_id} results: ", metrics_results)
            print("======================================================")
            print()

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")

    save_data = {}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results
    save_data["is_lora"] = args.lora
    save_data["base_model"] = args.base_model if args.lora else None

    # 确保结果目录存在
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)

    _save_metrics_files(args.results_file, mean_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test(args)
