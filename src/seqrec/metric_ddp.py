import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


def setup_distributed():
    """初始化分布式环境"""
    # torchrun 会自动设置 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK'
    # 'LOCAL_RANK' 是 torchrun 自动为每个进程设置的
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    print(
        f"[Rank {rank}] DDP setup: world_size={world_size}, local_rank={local_rank}"
    )
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def test(args: argparse.Namespace):
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)

    set_seed(args.seed)

    # 只在主进程打印参数
    if rank == 0:
        print(vars(args))

    # 使用load_model_for_inference加载模型
    if rank == 0:
        print("\n加载模型...")

    # ======================================================
    #  ⬇️  修改点 1: 传入 'device'
    # ======================================================
    model, processor = load_model_for_inference(
        model_type=args.model_type,
        ckpt_path=args.ckpt_path,
        use_lora=args.lora,
        model_path=args.base_model if args.lora else None,
        device=device,  # <-- 将当前进程的 device 传递下去
    )

    # ======================================================
    #  ⬇️  修改点 2: 移除多余的 .to(device)
    # ======================================================
    # 确保模型在正确的设备上
    # model.to(device) # <-- 移除此行 (load_model_for_inference 已处理)
    # ======================================================

    # 注意：对于纯推理，DDP (DistributedDataParallel) 包装不是必需的，
    # 只要每个GPU加载相同的模型并处理不同的数据批次即可。
    # 如果模型过大需要模型并行，则需要 FSDP 或 Deepspeed。
    # 这里我们假设是数据并行推理。

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
    test_data = load_test_dataset(args)
    all_items = test_data.get_all_items()

    if rank == 0:
        print("Num test data (total):", len(test_data))

    # DDP: 使用 DistributedSampler
    test_sampler = DistributedSampler(
        test_data, num_replicas=world_size, rank=rank, shuffle=True
    )

    if args.model_type == "llama":
        collator = TestCollator(args, tokenizer=processor)
        split_word = "Response: "  # 这个变量在原代码中定义了但未使用，保留
    elif args.model_type in ["qwen"]:
        collator = ChatTemplateTestCollator(args, tokenizer=processor)
        split_word = "assistant"  # 这个变量在原代码中定义了但未使用，保留
    else:
        collator = UnifiedTestCollator(args, processor_or_tokenizer=processor)
        split_word = "assistant"  # 这个变量在原代码中定义了但未使用，保留

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        # DDP: sampler 负责 shuffle，这里必须为 False
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        # DDP: 传入 sampler
        sampler=test_sampler,
    )

    if rank == 0:
        print(f"\n数据集大小 (total): {len(test_data)}")
        print(f"每个GPU的测试批次大小: {args.test_batch_size}")
        print(f"测试prompt IDs: {prompt_ids}")

    model.eval()

    # 解析评估指标
    metrics = args.metrics.split(",")
    all_prompt_results = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            if rank == 0:
                print(f"\n评估Prompt {prompt_id}...")

            # DDP: 确保每个 epoch 的 shuffle 不同
            test_loader.sampler.set_epoch(prompt_id)
            test_loader.dataset.set_prompt(prompt_id)

            # 存储 *本地* 进程的指标 *总和*
            local_metrics_sums = dict.fromkeys(metrics, 0.0)
            # 存储 *本地* 进程处理的样本总数
            local_total = 0

            # DDP: rank 0 才显示 tqdm
            iterable = tqdm(
                test_loader, desc=f"Prompt {prompt_id}", disable=(rank != 0)
            )

            for step, batch in enumerate(iterable):
                inputs = batch[0]
                targets = batch[1]
                batch_size = len(targets)
                local_total += batch_size

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

                output_text = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(
                    output_text,
                    scores,
                    targets,
                    args.num_beams,
                    all_items=all_items if args.filter_items else None,
                )

                # 这是批次的指标 *平均值*
                batch_metrics_res = get_metrics_results(topk_res, metrics)

                # 累加 *总和* (平均值 * 批次大小)
                for m, res in batch_metrics_res.items():
                    local_metrics_sums[m] += res

                if (step + 1) % 10 == 0 and rank == 0:
                    temp = {}
                    for m in local_metrics_sums:
                        if local_total > 0:
                            temp[m] = local_metrics_sums[m] / local_total
                    print(f" (Rank 0) Step {step + 1} temp results: {temp}")

            # --- DDP 聚合 ---
            # 1. 聚合样本总数
            total_tensor = torch.tensor(
                local_total, dtype=torch.long, device=device
            )
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            global_total = total_tensor.item()

            # 2. 聚合每个指标的总和
            global_metrics_results = {}
            for m in metrics:
                metric_sum_tensor = torch.tensor(
                    local_metrics_sums[m], dtype=torch.float64, device=device
                )
                dist.all_reduce(metric_sum_tensor, op=dist.ReduceOp.SUM)

                # 计算全局平均值 (仅 rank 0 需要)
                if rank == 0:
                    if global_total > 0:
                        global_metrics_results[m] = (
                            metric_sum_tensor.item() / global_total
                        )
                    else:
                        global_metrics_results[m] = 0.0

            # --- DDP 聚合结束 ---

            # 只有主进程 (rank 0) 负责记录和打印
            if rank == 0:
                all_prompt_results.append(global_metrics_results)
                print("======================================================")
                print(args.ckpt_path)
                print("======================================================")
                print(
                    f"Prompt {prompt_id} global results: ",
                    global_metrics_results,
                )
                print(f"(Based on {global_total} total samples)")
                print("======================================================")
                print()

            # 同步所有进程，确保 rank 0 已经处理完再进入下一个 prompt
            dist.barrier()

    # 只有主进程 (rank 0) 负责计算最终结果和保存
    if rank == 0:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results if m in _]
            if all_res:
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
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)

        print(f"Results saved to {args.results_file}")

    # 清理
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    # 启动 DDP 测试
    # test(args) 在内部处理 DDP setup/cleanup
    # try...finally 不是必需的，因为 torchrun 会管理进程
    # 但加上 cleanup_distributed 的调用是个好习惯
    try:
        test(args)
    except Exception as e:
        print(f"Error occurred: {e}")
        # 确保即使出错也尝试清理
        if dist.is_initialized():
            cleanup_distributed()
