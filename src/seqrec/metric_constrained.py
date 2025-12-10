import argparse
import json
import os
from typing import Callable, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LogitsProcessorList

from src.collator import (
    ChatTemplateTestCollator,
    TestCollator,
    UnifiedTestCollator,
)
from src.data_rl import FusionSeqRecDataset, SeqRecDataset
from src.evaluate import get_metrics_results, get_topk_results
from src.parser import parse_dataset_args, parse_global_args, parse_test_args
from src.prompt import all_prompt
from src.rl.LogitProcessor import ConstrainedLogitsProcessor
from src.utils import load_model_for_inference, set_seed


def load_test_dataset_rl(args: argparse.Namespace, logger=None, local_rank=0):
    """加载 data_rl 中实现的测试数据集，目前支持 seqrec 和 fusionseqrec。"""

    dataset_list = args.dataset.split(",")
    test_task = args.test_task.lower()
    eval_split = getattr(args, "eval_split", "test").lower()
    if eval_split not in {"test", "valid"}:
        raise ValueError(
            f"Unsupported eval_split '{eval_split}'. Choose from ['test', 'valid']."
        )
    test_data = None

    for dataset in dataset_list:
        if test_task == "seqrec":
            test_data = SeqRecDataset(
                args,
                mode=eval_split,
                dataset=dataset,
                sample_num=args.sample_num,
                logger=logger,
                local_rank=local_rank,
            )
        elif test_task == "fusionseqrec":
            test_data = FusionSeqRecDataset(
                args,
                mode=eval_split,
                dataset=dataset,
                sample_num=args.sample_num,
                logger=logger,
                local_rank=local_rank,
            )
        else:
            error_string = f"Unsupported task {args.test_task} for constrained metric"
            raise NotImplementedError(error_string)

    if test_data is None:
        raise RuntimeError("No dataset constructed. Check --dataset setting.")

    return test_data


def infer_prefix_index(base_model: str) -> int:
    base_lower = (base_model or "").lower()
    if "llava" in base_lower:
        return 7
    if "gpt2" in base_lower:
        return 4
    return 3


def build_prefix_allowed_tokens_fn(hash_dict: Dict[str, List[int]]) -> Callable:
    def get_hash(x) -> str:
        if isinstance(x, torch.Tensor):
            seq = x.tolist()
        else:
            seq = list(x)
        return "-".join(str(_) for _ in seq)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        return hash_dict.get(hash_number, [])

    return prefix_allowed_tokens_fn


def test(args: argparse.Namespace):
    set_seed(args.seed)
    print(vars(args))

    eval_split = getattr(args, "eval_split", "test")
    device = torch.device("cuda", args.gpu_id)

    print("\n加载模型...")
    model, processor = load_model_for_inference(
        model_type=args.model_type,
        ckpt_path=args.ckpt_path,
        use_lora=args.lora,
        model_path=args.base_model if args.lora else None,
    )

    if not hasattr(model, "device"):
        model.to(device)

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

    test_data = load_test_dataset_rl(args)
    all_items = test_data.get_all_items()
    print(f"Num {eval_split} data:", len(test_data))

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
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\n{eval_split} 数据集大小: {len(test_data)}")
    print(f"测试批次大小: {args.test_batch_size}")
    print(f"测试prompt IDs: {prompt_ids}")

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []

    prefix_index = infer_prefix_index(args.base_model)
    hash_dict = test_data.build_hash_dict(tokenizer, prefix_index=prefix_index)
    print(
        f"构建 hash_dict 完成，共 {len(hash_dict)} 条，使用 prefix_index={prefix_index}"
    )
    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(hash_dict)

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
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % 10 == 0:
                    temp = {m: metrics_results[m] / total for m in metrics_results}
                    print(temp)

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / max(total, 1)

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
    save_data["eval_split"] = eval_split

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test(args)
