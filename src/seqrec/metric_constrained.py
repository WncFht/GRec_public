import argparse
import json
import os
import time
from collections.abc import Callable
from typing import Any

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
from src.evaluate import (
    clean_predictions,
    get_metrics_results,
    get_topk_results,
)
from src.parser import parse_dataset_args, parse_global_args, parse_test_args
from src.prompt import all_prompt
from src.rl.LogitProcessor import ConstrainedLogitsProcessor
from src.utils import load_model_for_inference, set_seed


class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, Any]:
        return index, self.dataset[index]

    def set_prompt(self, prompt_id: int) -> None:
        if hasattr(self.dataset, "set_prompt"):
            self.dataset.set_prompt(prompt_id)

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


def _make_indexed_collator(base_collator):
    def collate(batch):
        indices, samples = zip(*batch, strict=False)
        collated = base_collator(list(samples))
        if not isinstance(collated, tuple) or len(collated) < 2:
            raise ValueError(
                "Base collator must return a tuple (inputs, targets, ...)"
            )
        return (collated[0], collated[1], list(indices), *collated[2:])

    return collate


def _atomic_json_dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _rollout_key_from_args(args: argparse.Namespace, eval_split: str) -> dict:
    return {
        "ckpt_path": args.ckpt_path,
        "model_type": args.model_type,
        "lora": bool(args.lora),
        "base_model": args.base_model,
        "dataset": args.dataset,
        "data_path": args.data_path,
        "index_file": args.index_file,
        "ratio_dataset": args.ratio_dataset,
        "test_task": str(args.test_task).lower(),
        "eval_split": str(eval_split).lower(),
        "sample_num": int(getattr(args, "sample_num", -1)),
        "num_beams": int(args.num_beams),
        "max_new_tokens": int(args.max_new_tokens),
    }


def _load_rollout_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _validate_rollout_cache(
    cache: dict,
    args: argparse.Namespace,
    eval_split: str,
    prompt_ids: list[int],
) -> tuple[bool, str]:
    if not isinstance(cache, dict):
        return False, "cache is not a dict"
    if cache.get("schema_version") != 1:
        return False, "unsupported schema_version"
    meta = cache.get("meta", {})
    key = meta.get("key", {})
    expected_key = _rollout_key_from_args(args, eval_split)
    if key != expected_key:
        return False, "meta.key mismatch"
    if "targets" not in cache or not isinstance(cache["targets"], list):
        return False, "missing targets"
    prompts = cache.get("prompts", {})
    if not isinstance(prompts, dict):
        return False, "prompts is not a dict"
    missing = [pid for pid in prompt_ids if str(pid) not in prompts]
    if missing:
        return False, f"missing prompts {missing}"
    num_samples = len(cache["targets"])
    for pid in prompt_ids:
        pdata = prompts[str(pid)]
        preds = pdata.get("predictions")
        scores = pdata.get("scores")
        if not isinstance(preds, list) or len(preds) != num_samples:
            return False, f"predictions length mismatch for prompt {pid}"
        if not isinstance(scores, list) or len(scores) != num_samples:
            return False, f"scores length mismatch for prompt {pid}"
    return True, "ok"


def _parse_prompt_ids(test_prompt_ids: str) -> list[int]:
    if test_prompt_ids == "all":
        return list(range(len(all_prompt["seqrec"])))
    return [int(_) for _ in test_prompt_ids.split(",") if str(_).strip()]


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
            error_string = (
                f"Unsupported task {args.test_task} for constrained metric"
            )
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


def build_prefix_allowed_tokens_fn(hash_dict: dict[str, list[int]]) -> Callable:
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
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False))
    print(vars(args))

    eval_split = getattr(args, "eval_split", "test")
    eval_split_lower = str(eval_split).lower()

    prompt_ids = _parse_prompt_ids(args.test_prompt_ids)

    rollout_file = getattr(args, "rollout_file", "")
    force_rollout = bool(getattr(args, "force_rollout", False))
    skip_rollout = bool(getattr(args, "skip_rollout", False))

    if rollout_file:
        if os.path.exists(rollout_file) and not force_rollout:
            try:
                cache = _load_rollout_cache(rollout_file)
                ok, reason = _validate_rollout_cache(
                    cache, args, eval_split_lower, prompt_ids
                )
                if ok:
                    print(f"使用缓存 rollout 结果: {rollout_file}")
                    all_items = None
                    if args.filter_items:
                        test_data = load_test_dataset_rl(args)
                        all_items = test_data.get_all_items()

                    metrics = args.metrics.split(",")
                    all_prompt_results = []
                    for prompt_id in prompt_ids:
                        pdata = cache["prompts"][str(prompt_id)]
                        targets = cache["targets"]
                        predictions = pdata["predictions"]
                        scores = pdata["scores"]

                        preds_flat = [p for row in predictions for p in row]
                        scores_flat = [s for row in scores for s in row]

                        topk_res = get_topk_results(
                            preds_flat,
                            scores_flat,
                            targets,
                            args.num_beams,
                            all_items=all_items if args.filter_items else None,
                        )

                        metrics_sum = get_metrics_results(topk_res, metrics)
                        total = max(len(targets), 1)
                        metrics_results = {
                            m: metrics_sum[m] / total for m in metrics_sum
                        }

                        all_prompt_results.append(metrics_results)
                        print(
                            "======================================================"
                        )
                        print(args.ckpt_path)
                        print(
                            "======================================================"
                        )
                        print(
                            f"Prompt {prompt_id} cached constrained results: ",
                            metrics_results,
                        )
                        print(f"(Based on {len(targets)} total samples)")
                        print(
                            "======================================================"
                        )
                        print()

                    mean_results = {}
                    min_results = {}
                    max_results = {}
                    for m in metrics:
                        all_res = [_[m] for _ in all_prompt_results if m in _]
                        if all_res:
                            mean_results[m] = sum(all_res) / len(all_res)
                            min_results[m] = min(all_res)
                            max_results[m] = max(all_res)

                    print(
                        "======================================================"
                    )
                    print("Mean results: ", mean_results)
                    print("Min results: ", min_results)
                    print("Max results: ", max_results)
                    print(
                        "======================================================"
                    )

                    save_data = {
                        "test_prompt_ids": args.test_prompt_ids,
                        "mean_results": mean_results,
                        "min_results": min_results,
                        "max_results": max_results,
                        "all_prompt_results": all_prompt_results,
                        "is_lora": args.lora,
                        "base_model": args.base_model if args.lora else None,
                        "eval_split": eval_split_lower,
                        "rollout_file": rollout_file,
                        "rollout_cached": True,
                    }

                    os.makedirs(
                        os.path.dirname(args.results_file) or ".", exist_ok=True
                    )
                    with open(args.results_file, "w") as f:
                        json.dump(save_data, f, indent=4, ensure_ascii=False)

                    return
                if skip_rollout:
                    raise ValueError(f"rollout cache invalid: {reason}")
                print(
                    f"rollout 缓存无效({reason})，将重新 rollout 并覆盖: {rollout_file}"
                )
            except Exception as exc:
                if skip_rollout:
                    raise
                print(f"读取 rollout 缓存失败，将重新 rollout: {exc}")
        elif skip_rollout and not os.path.exists(rollout_file):
            raise FileNotFoundError(
                f"--skip_rollout 但未找到 rollout_file: {rollout_file}"
            )
    elif skip_rollout:
        raise ValueError("--skip_rollout 需要同时设置 --rollout_file")

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

    test_data = load_test_dataset_rl(args)
    all_items = test_data.get_all_items()
    print(f"Num {eval_split} data:", len(test_data))

    if args.model_type == "llama":
        collator = TestCollator(args, tokenizer=processor)
    elif args.model_type in ["qwen"]:
        collator = ChatTemplateTestCollator(args, tokenizer=processor)
    else:
        collator = UnifiedTestCollator(args, processor_or_tokenizer=processor)

    dataset_for_loader = (
        _IndexedDataset(test_data) if rollout_file else test_data
    )
    collate_fn = _make_indexed_collator(collator) if rollout_file else collator

    test_loader = DataLoader(
        dataset_for_loader,
        batch_size=args.test_batch_size,
        collate_fn=collate_fn,
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
        targets_cache: list[str | None] | None = None
        prompts_cache: dict[str, dict[str, list]] = {}
        if rollout_file:
            targets_cache = [None] * len(test_data)

        for prompt_id in prompt_ids:
            print(f"\n评估Prompt {prompt_id}...")
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0
            predictions_cache: list[list[str] | None] | None = None
            scores_cache: list[list[float] | None] | None = None
            if rollout_file:
                predictions_cache = [None] * len(test_data)
                scores_cache = [None] * len(test_data)

            for step, batch in enumerate(
                tqdm(test_loader, desc=f"Prompt {prompt_id}")
            ):
                inputs = batch[0]
                targets = batch[1]
                indices = batch[2] if rollout_file else None
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
                pred_items = clean_predictions(output_text)

                if rollout_file and indices is not None:
                    assert (
                        predictions_cache is not None
                        and scores_cache is not None
                        and targets_cache is not None
                    )
                    scores_list = scores.detach().cpu().tolist()
                    for i, dataset_index in enumerate(indices):
                        start = i * args.num_beams
                        end = (i + 1) * args.num_beams
                        predictions_cache[dataset_index] = pred_items[start:end]
                        scores_cache[dataset_index] = scores_list[start:end]
                        if targets_cache[dataset_index] is None:
                            targets_cache[dataset_index] = targets[i]

                topk_res = get_topk_results(
                    pred_items,
                    scores,
                    targets,
                    args.num_beams,
                    all_items=all_items if args.filter_items else None,
                    clean=False,
                )

                batch_metrics_res = get_metrics_results(topk_res, metrics)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % 10 == 0:
                    temp = {
                        m: metrics_results[m] / total for m in metrics_results
                    }
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

            if rollout_file:
                assert (
                    predictions_cache is not None
                    and scores_cache is not None
                    and targets_cache is not None
                )
                if any(v is None for v in predictions_cache):
                    raise RuntimeError(
                        f"rollout predictions 缺失，prompt_id={prompt_id}"
                    )
                if any(v is None for v in scores_cache):
                    raise RuntimeError(
                        f"rollout scores 缺失，prompt_id={prompt_id}"
                    )
                prompts_cache[str(prompt_id)] = {
                    "predictions": predictions_cache,
                    "scores": scores_cache,
                }

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

    if rollout_file:
        assert targets_cache is not None
        if any(v is None for v in targets_cache):
            raise RuntimeError("rollout targets 缺失")
        rollout_payload = {
            "schema_version": 1,
            "meta": {
                "created_at": int(time.time()),
                "key": _rollout_key_from_args(args, eval_split_lower),
            },
            "targets": targets_cache,
            "prompts": prompts_cache,
        }
        _atomic_json_dump(rollout_file, rollout_payload)
        save_data["rollout_file"] = rollout_file
        save_data["rollout_cached"] = False
        print(f"Rollout saved to {rollout_file}")

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    rollout_args = parser.add_argument_group("rollout_args")
    rollout_args.add_argument(
        "--rollout_file",
        type=str,
        default="",
        help="Rollout cache path (json). If exists, will reuse unless --force_rollout.",
    )
    rollout_args.add_argument(
        "--force_rollout",
        action="store_true",
        default=False,
        help="Force rerun rollout and overwrite rollout_file even if it exists.",
    )
    rollout_args.add_argument(
        "--skip_rollout",
        action="store_true",
        default=False,
        help="Only compute metrics from rollout_file; error if missing/invalid.",
    )

    args = parser.parse_args()
    test(args)
