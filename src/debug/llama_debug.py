import argparse
import json
import os

import torch
from collator import TestCollator
from evaluate import get_metrics_results, get_topk_results
from peft import PeftModel
from prompt import all_prompt
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils import (
    load_test_dataset,
    parse_dataset_args,
    parse_global_args,
    parse_test_args,
    set_seed,
)


def test(args):
    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)

    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path)
    if args.lora:
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    # assert model.config.vocab_size == len(tokenizer)

    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print("data num:", len(test_data))

    model.eval()

    # --- 模仿 debug_test.py 的主要调试逻辑 ---

    # 1. 选择一个 prompt进行测试
    prompt_id_to_test = int(args.test_prompt_ids.split(",")[0])
    print(f"\n--- 将使用 Prompt ID: {prompt_id_to_test} 进行调试 ---")

    # 2. 为数据集设置指定的 prompt 并获取一个样本批次
    test_loader.dataset.set_prompt(prompt_id_to_test)
    try:
        sample_batch = next(iter(test_loader))
    except StopIteration:
        print("错误: 测试数据加载器为空，无法获取样本。")
        return

    print(f"\n--- 开始调试单个批次 (Batch Size={args.test_batch_size}) ---")

    with torch.no_grad():
        inputs, targets = sample_batch

        # 3. 打印输入信息
        print("\n[1. 输入信息]")
        print(f"  - 批次大小 (Batch Size): {len(targets)}")
        print(f"  - 真实目标 (Targets): {targets}")
        # 解码并打印批次中第一个样本的文本输入
        first_sample_text = tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        print(f"  - 第一个样本的文本输入 (解码后):\n'{first_sample_text}'")

        # 将输入移动到设备
        inputs = inputs.to(device)

        # 4. 模型生成
        print("\n[2. 模型生成]")
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
        )

        output_ids = output["sequences"]
        scores = output["sequences_scores"]

        # 5. 打印原始输出
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
                clean_pred = pred.split("### Response:")[-1].strip()
                print(
                    f"      - Beam {j + 1}: '{clean_pred}' (Score: {score:.4f})"
                )

        # 6. 评估计算
        print("\n[3. 评估计算]")
        topk_res = get_topk_results(
            output_texts,
            scores.cpu().tolist(),
            targets,
            args.num_beams,
            all_items=all_items if args.filter_items else None,
        )
        print("  - `get_topk_results` 的输出 (命中列表):")
        for i, res in enumerate(topk_res):
            print(f"    - 样本 {i + 1} (目标: {targets[i]}): {res}")

        metrics_to_calc = args.metrics.split(",")
        batch_metrics_res = get_metrics_results(topk_res, metrics_to_calc)
        print("\n  - `get_metrics_results` 的输出 (批次指标总和):")
        print(f"    {json.dumps(batch_metrics_res, indent=4)}")

    print("\n--- 调试结束 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec Debug Script")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    # 添加或覆盖部分参数以方便调试
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="生成的最大新 token 数量",
    )

    args = parser.parse_args()
    test(args)
