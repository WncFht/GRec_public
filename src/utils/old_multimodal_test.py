# multimodal_test.py
import argparse
import json
import os

import torch
from collator import UnifiedTestCollator
from evaluate import get_metrics_results, get_topk_results
from peft import PeftModel
from prompt import all_prompt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from utils import *


def load_model_with_new_tokens(args, device_map):
    """
    加载带有新增tokens的多模态模型
    """
    # 1. 加载processor
    ckpt_path = os.path.abspath(args.ckpt_path)
    processor = AutoProcessor.from_pretrained(
        ckpt_path, trust_remote_code=True, local_files_only=True
    )
    # 2. 读取token元信息
    token_meta_path = os.path.join(ckpt_path, "token_meta.json")
    if os.path.exists(token_meta_path):
        with open(token_meta_path) as f:
            token_meta = json.load(f)
        original_vocab_size = token_meta["original_vocab_size"]
        new_vocab_size = token_meta["new_vocab_size"]
        print(
            f"从元信息加载: 原始vocab_size={original_vocab_size}, 新vocab_size={new_vocab_size}"
        )
    else:
        print("警告: 未找到token_meta.json，假设没有新增tokens")
        original_vocab_size = len(processor.tokenizer)
        new_vocab_size = original_vocab_size

    # 3. 加载基础模型
    if args.lora:
        print("加载基础模型用于LoRA...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )

        # 调整词汇表大小以匹配训练时的设置
        if new_vocab_size > original_vocab_size:
            model.resize_token_embeddings(new_vocab_size)
            model.config.vocab_size = new_vocab_size

        print("加载LoRA适配器...")
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

        # 验证词汇表大小
        assert (
            len(processor.tokenizer) == new_vocab_size
        ), f"Tokenizer size {len(processor.tokenizer)} != model vocab size {new_vocab_size}"

    else:
        print("加载完整微调模型...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )

    print(f"模型加载完成，词汇表大小: {len(processor.tokenizer)}")
    return model, processor


def test(args):
    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)

    # 加载模型和processor
    model, processor = load_model_with_new_tokens(args, device_map)
    processor.tokenizer.padding_side = "left"

    # 确定测试的prompt IDs
    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
        elif args.test_task.lower() == "mmitem2index":
            prompt_ids = range(len(all_prompt["mmitem2index"]))
        elif args.test_task.lower() == "mmindex2item":
            prompt_ids = range(len(all_prompt["mmindex2item"]))
        else:
            raise ValueError(f"不支持的测试任务: {args.test_task}")
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    # 加载测试数据
    test_data = load_test_dataset(args)
    collator = UnifiedTestCollator(
        args, processor
    )  # 序列推荐评估不需要多模态数据
    all_items = test_data.get_all_items()

    # 获取前缀允许的tokens函数（用于约束生成）
    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(
        processor.tokenizer
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,  # 测试时不需要shuffle
        num_workers=0,  # 多模态数据加载时避免多进程问题
        pin_memory=True,
    )

    print("测试数据数量:", len(test_data))

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            print(f"\n开始测试 Prompt {prompt_id}...")

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(
                tqdm(test_loader, desc=f"Prompt {prompt_id}")
            ):
                inputs = batch[0]
                targets = batch[1]
                total += len(targets)

                # 将inputs移动到设备
                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                # 生成输出
                # try:
                output = model.generate(
                    **inputs,
                    # max_new_tokens=args.max_new_tokens,
                    max_new_tokens=100,
                    # prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    prefix_allowed_tokens_fn=None,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
                # except Exception as e:
                #     print(f"生成时出错: {e}")
                #     print(f"输入形状: {inputs['input_ids'].shape if 'input_ids' in inputs else 'No input_ids'}")
                #     continue

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                # 解码输出
                output_texts = processor.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                import pdb

                pdb.set_trace()

                # 获取topk结果
                topk_res = get_topk_results(
                    output_texts,
                    scores,
                    targets,
                    args.num_beams,
                    all_items=all_items if args.filter_items else None,
                )

                # 计算批次指标
                batch_metrics_res = get_metrics_results(topk_res, metrics)

                # 累积指标
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                # 定期打印中间结果
                if (step + 1) % args.print_freq == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(f"Step {step + 1}, 当前指标: {temp}")

            # 计算平均指标
            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            print("=" * 60)
            print(f"Prompt {prompt_id} 结果: {metrics_results}")
            print("=" * 60)
            print("")

    # 计算所有prompt的统计结果
    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("=" * 60)
    print("平均结果: ", mean_results)
    print("最小结果: ", min_results)
    print("最大结果: ", max_results)
    print("=" * 60)

    # 保存结果
    save_data = {
        "test_task": args.test_task,
        "test_prompt_ids": args.test_prompt_ids,
        "mean_results": mean_results,
        "min_results": min_results,
        "max_results": max_results,
        "all_prompt_results": all_prompt_results,
        "model_info": {
            "base_model": args.base_model,
            "ckpt_path": args.ckpt_path,
            "lora": args.lora,
            "vocab_size": len(processor.tokenizer),
        },
        "test_config": {
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "test_batch_size": args.test_batch_size,
            "use_constrained_generation": args.use_constrained_generation,
        },
    }

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"结果已保存到: {args.results_file}")


# def parse_test_args(parser):
#     """解析测试相关参数"""
#     parser.add_argument('--ckpt_path', type=str, required=True,
#                     help='Model checkpoint path')
#     parser.add_argument('--base_model', type=str, required=True,
#                     help='Base model path (for LoRA)')
#     parser.add_argument('--lora', action='store_true',
#                     help='Whether to use LoRA model')
#     parser.add_argument('--gpu_id', type=int, default=0,
#                     help='GPU ID')
#     parser.add_argument('--test_task', type=str, default='seqrec',
#                     choices=['seqrec', 'itemsearch', 'fusionseqrec', 'mmitem2index', 'mmindex2item'],
#                     help='Test task type')
#     parser.add_argument('--test_prompt_ids', type=str, default='all',
#                     help='Test prompt IDs, comma-separated or "all"')
#     parser.add_argument('--test_batch_size', type=int, default=1,
#                     help='Test batch size')
#     parser.add_argument('--num_beams', type=int, default=5,
#                     help='Number of beams for beam search')
#     parser.add_argument('--max_new_tokens', type=int, default=10,
#                     help='Maximum number of new tokens to generate')
#     parser.add_argument('--metrics', type=str, default='hit@1,hit@5,hit@10,ndcg@5,ndcg@10',
#                     help='Evaluation metrics, comma-separated')
#     parser.add_argument('--results_file', type=str, default='results/test_results.json',
#                     help='Results file save path')
#     parser.add_argument('--filter_items', action='store_true',
#                     help='Whether to filter items not in candidate set')
#     parser.add_argument('--print_freq', type=int, default=10,
#                     help='Print frequency')

#     return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiModal Recommendation Model Test"
    )
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser.add_argument(
        "--use_constrained_generation",
        action="store_true",
        help="Whether to use constrained generation",
    )  # 物品序列推荐使用，文本生成不用
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4,
        help="Maximum number of new tokens to generate",
    )  # 物品序列推荐只生成4个token ids
    parser.add_argument(
        "--print_freq", type=int, default=4, help="Print frequency"
    )

    args = parser.parse_args()
    test(args)
