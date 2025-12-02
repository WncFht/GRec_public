import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import LogitsProcessorList

from src.collator import (
    ChatTemplateTestCollator,
    TestCollator,
    UnifiedTestCollator,
)
from src.evaluate import get_metrics_results, get_topk_results
from src.parser import parse_dataset_args, parse_global_args, parse_test_args
from src.prompt import all_prompt
from src.rl.LogitProcessor import ConstrainedLogitsProcessor
from src.seqrec.metric_constrained import (
    build_prefix_allowed_tokens_fn,
    infer_prefix_index,
    load_test_dataset_rl,
)
from src.utils import load_model_for_inference, set_seed


def setup_distributed():
    """Initialize default NCCL process group and set CUDA device."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    print(
        f"[Rank {rank}] DDP setup complete: world_size={world_size}, local_rank={local_rank}"
    )
    return rank, world_size, local_rank


def cleanup_distributed():
    """Destroy process group when evaluation finishes or errors out."""
    if dist.is_initialized():
        dist.destroy_process_group()


def test(args: argparse.Namespace):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)

    set_seed(args.seed)

    if rank == 0:
        print(vars(args))
        print("\n加载模型...")

    model, processor = load_model_for_inference(
        model_type=args.model_type,
        ckpt_path=args.ckpt_path,
        use_lora=args.lora,
        model_path=args.base_model if args.lora else None,
        device=device,
    )

    tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.test_prompt_ids == "all":
        prompt_ids = range(len(all_prompt["seqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset_rl(args, local_rank=rank)
    all_items = test_data.get_all_items()

    if rank == 0:
        print("Num test data (total):", len(test_data))

    test_sampler = DistributedSampler(
        test_data, num_replicas=world_size, rank=rank, shuffle=True
    )

    if args.model_type == "llama":
        collator = TestCollator(args, tokenizer=processor)
    elif args.model_type in ["qwen"]:
        collator = ChatTemplateTestCollator(args, tokenizer=processor)
    else:
        collator = UnifiedTestCollator(args, processor_or_tokenizer=processor)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=test_sampler,
    )

    if rank == 0:
        print(f"\n数据集大小 (total): {len(test_data)}")
        print(f"每个GPU的测试批次大小: {args.test_batch_size}")
        print(f"测试prompt IDs: {prompt_ids}")

    prefix_index = infer_prefix_index(args.base_model)
    hash_dict = test_data.build_hash_dict(tokenizer, prefix_index=prefix_index)
    if rank == 0:
        print(
            f"构建 hash_dict 完成，共 {len(hash_dict)} 条，使用 prefix_index={prefix_index}"
        )
    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(hash_dict)

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            if rank == 0:
                print(f"\n评估Prompt {prompt_id}...")

            test_loader.sampler.set_epoch(prompt_id)
            test_loader.dataset.set_prompt(prompt_id)

            local_metrics_sums = dict.fromkeys(metrics, 0.0)
            local_total = 0

            iterable = tqdm(
                test_loader, desc=f"Prompt {prompt_id}", disable=(rank != 0)
            )

            for step, batch in enumerate(iterable):
                inputs = batch[0]
                targets = batch[1]
                batch_size = len(targets)
                local_total += batch_size

                inputs = {k: v.to(device) for k, v in inputs.items()}

                constrained_processor = ConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=args.num_beams,
                    base_model=args.base_model,
                    prefix_index=prefix_index,
                )
                logits_processor = LogitsProcessorList([constrained_processor])

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    temperature=1.0,
                    logits_processor=logits_processor,
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

                batch_metrics_res = get_metrics_results(topk_res, metrics)

                for m, res in batch_metrics_res.items():
                    local_metrics_sums[m] += res

                if (step + 1) % 10 == 0 and rank == 0:
                    temp = {}
                    for m in local_metrics_sums:
                        if local_total > 0:
                            temp[m] = local_metrics_sums[m] / local_total
                    print(f"(Rank 0) Step {step + 1} temp results: {temp}")

            total_tensor = torch.tensor(
                local_total, dtype=torch.long, device=device
            )
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            global_total = total_tensor.item()

            global_metrics_results = {}
            for m in metrics:
                metric_sum_tensor = torch.tensor(
                    local_metrics_sums[m], dtype=torch.float64, device=device
                )
                dist.all_reduce(metric_sum_tensor, op=dist.ReduceOp.SUM)

                if rank == 0:
                    if global_total > 0:
                        global_metrics_results[m] = (
                            metric_sum_tensor.item() / global_total
                        )
                    else:
                        global_metrics_results[m] = 0.0

            if rank == 0:
                all_prompt_results.append(global_metrics_results)
                print("======================================================")
                print(args.ckpt_path)
                print("======================================================")
                print(
                    f"Prompt {prompt_id} global constrained results: ",
                    global_metrics_results,
                )
                print(f"(Based on {global_total} total samples)")
                print("======================================================")
                print()

            dist.barrier()

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

        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)

        print(f"Results saved to {args.results_file}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    try:
        test(args)
    except Exception as exc:
        print(f"Error occurred: {exc}")
        cleanup_distributed()
