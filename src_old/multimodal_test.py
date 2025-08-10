# multimodal_test.py
import argparse
import json
import os
from collections.abc import Callable

import torch
from collator import UnifiedTestCollator
from evaluate import get_metrics_results, get_topk_results
from peft import PeftModel
from prompt import all_prompt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    T5ForConditionalGeneration,
)
from utils import (
    load_test_dataset,
    parse_dataset_args,
    parse_global_args,
    parse_test_args,
    set_seed,
)


def get_tokenizer(
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
) -> AutoTokenizer:
    """从 Processor 或 Tokenizer 中获取 Tokenizer"""
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def load_model_with_new_tokens(
    args: argparse.Namespace, device_map: dict[str, int]
) -> tuple[
    Qwen2_5_VLForConditionalGeneration | T5ForConditionalGeneration,
    AutoProcessor | AutoTokenizer,
]:
    """
    加载带有新增tokens的多模态或单模态模型.

    Args:
    ----
        args (argparse.Namespace): 包含模型路径和LoRA配置的参数.
        device_map (dict[str, int]): 设备映射.

    Returns:
    -------
        tuple[Union[Qwen2_5_VLForConditionalGeneration, T5ForConditionalGeneration], Union[AutoProcessor, AutoTokenizer]]:
            加载的模型和处理器/tokenizer.

    """
    ckpt_path = os.path.abspath(args.ckpt_path)

    if args.model_type == "qwen_vl":
        # 1. 加载processor
        tokenizer_or_processor = AutoProcessor.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )
        tokenizer = get_tokenizer(tokenizer_or_processor)

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
            original_vocab_size = len(tokenizer)
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
                print("完成 model_embedding resize")

            print("加载LoRA适配器...")
            model = PeftModel.from_pretrained(
                model,
                args.ckpt_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
        else:
            print("加载完整微调模型...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.ckpt_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=device_map,
                trust_remote_code=True,
            )

    elif args.model_type == "t5":
        print("加载 T5 模型...")
        tokenizer_or_processor = AutoTokenizer.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )
        tokenizer = get_tokenizer(tokenizer_or_processor)

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
            original_vocab_size = len(tokenizer)
            new_vocab_size = original_vocab_size

        if args.lora:
            print("加载基础模型用于LoRA...")
            model = T5ForConditionalGeneration.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=device_map,
                trust_remote_code=True,
            )
            if new_vocab_size > original_vocab_size:
                model.resize_token_embeddings(new_vocab_size)
                model.config.vocab_size = new_vocab_size
                print("完成 model_embedding resize")

            print("加载LoRA适配器...")
            model = PeftModel.from_pretrained(
                model,
                args.ckpt_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
        else:
            print("加载完整微调的 T5 模型...")
            model = T5ForConditionalGeneration.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=device_map,
                trust_remote_code=True,
            )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    # 验证词汇表大小
    final_vocab_size = model.get_input_embeddings().weight.shape[0]
    assert (
        len(tokenizer) == final_vocab_size
    ), f"Tokenizer size {len(tokenizer)} != model vocab size {final_vocab_size}"

    print(f"模型加载完成，词汇表大小: {len(tokenizer)}")
    return model, tokenizer_or_processor


def setup_test(
    args: argparse.Namespace,
) -> tuple[
    Qwen2_5_VLForConditionalGeneration | T5ForConditionalGeneration,
    AutoProcessor | AutoTokenizer,
    torch.device,
]:
    """
    初始化测试环境.

    Args:
    ----
        args (argparse.Namespace): 包含种子、GPU ID等设置的参数.

    Returns:
    -------
        tuple[Union[Qwen2_5_VLForConditionalGeneration, T5ForConditionalGeneration], Union[AutoProcessor, AutoTokenizer], torch.device]:
            初始化后的模型、处理器/tokenizer和设备.

    """
    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)

    model, tokenizer_or_processor = load_model_with_new_tokens(args, device_map)
    tokenizer = get_tokenizer(tokenizer_or_processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer_or_processor, device


def get_prompt_ids(args: argparse.Namespace) -> list[int]:
    """
    获取测试的prompt ID列表.

    all_prompt 有 9 个 task:
    seqrec, itemsearch, fusionseqrec, mmitem2index, mmindex2item, textenrich,
    mmindex2item_v2, mmindex2item_v3, mmindex2item_v4

    Args:
    ----
        args (argparse.Namespace): 包含 test_prompt_ids 和 test_task 的参数.

    Returns:
    -------
        list[int]: 测试的prompt ID列表.

    """
    if args.test_prompt_ids == "all":
        task = args.test_task.lower()
        if task in all_prompt:
            return range(len(all_prompt[task]))
        raise ValueError(f"不支持的测试任务: {task}")
    # 挑具体的那个 test_task 的 test_prompt_ids 的 prompt
    return [int(_) for _ in args.test_prompt_ids.split(",")]


def prepare_test_data(
    args: argparse.Namespace,
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
) -> tuple[DataLoader, list, Callable]:
    """
    准备测试数据.

    Args:
    ----
        args (argparse.Namespace): 包含数据集配置的参数.
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
        batch_size=args.test_batch_size,
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
    model: Qwen2_5_VLForConditionalGeneration | T5ForConditionalGeneration,
    test_loader: DataLoader,
    device: torch.device,
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
    args: argparse.Namespace,
    all_items: list,
    metrics: list[str],
) -> dict[str, float]:
    """
    评估单个prompt.

    Args:
    ----
        prompt_id (int): prompt id.
        model (Union[Qwen2_5_VLForConditionalGeneration, T5ForConditionalGeneration]): 模型.
        test_loader (DataLoader): 测试数据.
        device (torch.device): 设备.
        tokenizer_or_processor (Union[AutoProcessor, AutoTokenizer]): 处理器或tokenizer.
        args (argparse.Namespace): 参数.
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
            if args.use_constrained_generation
            else None
        )

        output: dict[str, torch.Tensor] = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
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
            args.num_beams,
            all_items=all_items
            if "filter_items" in args and args.filter_items
            else None,
            model_type=args.model_type,
        )
        # 获得 metrics 结果
        batch_metrics_res = get_metrics_results(topk_res, metrics)

        # # 如果 某个样本的 topk_res 不全为0，则打印该样本的详细信息
        # for i, topk_sample_res in enumerate(topk_res):
        #     if sum(topk_sample_res) != 0:
        #         print_hit_details(
        #             index=i,
        #             topk_hit_info=topk_sample_res,
        #             inputs=inputs,
        #             targets=targets,
        #             output_texts=output_texts,
        #             scores=scores,
        #             processor=tokenizer_or_processor,
        #             num_beams=args.num_beams,
        #         )

        for m, res in batch_metrics_res.items():
            metrics_results[m] = metrics_results.get(m, 0) + res

        if (step + 1) % args.print_freq == 0:
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
    args: argparse.Namespace,
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
) -> None:
    """
    汇总并保存结果.

    Args:
    ----
        all_prompt_results (list[dict[str, float]]): 所有 prompt 的评测结果.
        metrics (list[str]): 评价指标列表.
        args (argparse.Namespace): 参数.
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

    save_data = {
        "test_task": args.test_task,
        "test_prompt_ids": args.test_prompt_ids,
        "mean_results": mean_results,
        "min_results": min_results,
        "max_results": max_results,
        "all_prompt_results": all_prompt_results,
        "model_info": {
            "model_type": args.model_type,
            "base_model": args.base_model,
            "ckpt_path": args.ckpt_path,
            "lora": args.lora,
            "vocab_size": len(get_tokenizer(tokenizer_or_processor)),
        },
        "test_config": {
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "test_batch_size": args.test_batch_size,
            "use_constrained_generation": args.use_constrained_generation,
        },
    }

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"结果已保存到: {args.results_file}")


def test(args: argparse.Namespace) -> None:
    """
    执行完整的测试流程.

    Args:
    ----
        args (argparse.Namespace): 包含所有测试配置的参数.

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
    metrics: list[str] = args.metrics.split(",")
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
