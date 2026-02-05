import argparse
import json
import os
import time

import torch
from src.collator import (
    ChatTemplateTestCollator,
    TestCollator,
    UnifiedTestCollator,
)
from src.evaluate import clean_predictions, get_metrics_results, get_topk_results
from src.parser import parse_dataset_args, parse_global_args, parse_test_args
from src.rl.LogitProcessor import ConstrainedLogitsProcessor
from src.seqrec.metric_constrained import (
    _atomic_json_dump,
    _IndexedDataset,
    _load_rollout_cache,
    _make_indexed_collator,
    _parse_prompt_ids,
    _rollout_key_from_args,
    _validate_rollout_cache,
    build_prefix_allowed_tokens_fn,
    infer_prefix_index,
    load_test_dataset_rl,
)
from src.utils import load_model_for_inference, set_seed
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import LogitsProcessorList


class _DistributedEvalSampler(Sampler[int]):
    """
    DDP evaluation sampler without padding/duplication.

    - When shuffle=True, shuffles deterministically by (seed + epoch).
    - Splits indices by slicing: indices[rank::world_size].
    """

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        return iter(indices[self.rank : n : self.num_replicas])

    def __len__(self) -> int:
        n = len(self.dataset)
        if n <= self.rank:
            return 0
        return (n - self.rank + self.num_replicas - 1) // self.num_replicas


def setup_process():
    """
    Lightweight torchrun setup without torch.distributed/NCCL.

    torchrun will set env vars: RANK, WORLD_SIZE, LOCAL_RANK.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}] setup: world_size={world_size}, local_rank={local_rank}")
    return rank, world_size, local_rank


def test(args: argparse.Namespace):
    rank, world_size, local_rank = setup_process()
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    set_seed(args.seed, deterministic=getattr(args, "deterministic", False))
    eval_split = getattr(args, "eval_split", "test")
    eval_split_lower = eval_split.lower()

    prompt_ids = _parse_prompt_ids(args.test_prompt_ids)

    rollout_file = getattr(args, "rollout_file", "")
    force_rollout = bool(getattr(args, "force_rollout", False))
    skip_rollout = bool(getattr(args, "skip_rollout", False))

    if rollout_file and os.path.exists(rollout_file) and not force_rollout:
        try:
            cache = _load_rollout_cache(rollout_file)
            ok, reason = _validate_rollout_cache(
                cache, args, eval_split_lower, prompt_ids
            )
            if ok:
                if rank == 0:
                    print(f"使用缓存 rollout 结果: {rollout_file}")

                    all_items = None
                    if args.filter_items:
                        test_data = load_test_dataset_rl(args, local_rank=rank)
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
                        print("======================================================")
                        print(args.ckpt_path)
                        print("======================================================")
                        print(
                            f"Prompt {prompt_id} cached constrained results: ",
                            metrics_results,
                        )
                        print(f"(Based on {len(targets)} total samples)")
                        print("======================================================")
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

                    print("======================================================")
                    print("Mean results: ", mean_results)
                    print("Min results: ", min_results)
                    print("Max results: ", max_results)
                    print("======================================================")

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
            if rank == 0:
                print(
                    f"rollout 缓存无效({reason})，将重新 rollout 并覆盖: {rollout_file}"
                )
        except Exception as exc:
            if skip_rollout:
                raise
            if rank == 0:
                print(f"读取 rollout 缓存失败，将重新 rollout: {exc}")
    elif skip_rollout and rollout_file and not os.path.exists(rollout_file):
        raise FileNotFoundError(f"--skip_rollout 但未找到 rollout_file: {rollout_file}")
    elif skip_rollout and not rollout_file:
        raise ValueError("--skip_rollout 需要同时设置 --rollout_file")
    elif not rollout_file:
        raise ValueError(
            "This entrypoint requires --rollout_file for file-based merge/calc."
        )

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

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_data = load_test_dataset_rl(args, local_rank=rank)
    all_items = test_data.get_all_items()

    if rank == 0:
        print(f"Num {eval_split_lower} data (total):", len(test_data))

    dataset_for_loader = _IndexedDataset(test_data) if rollout_file else test_data
    test_sampler = _DistributedEvalSampler(
        dataset_for_loader,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )

    if args.model_type in ["qwen", "qwen2", "qwen2_5", "llama"]:
        collator = TestCollator(args, tokenizer=processor)
    elif args.model_type in ["qwen2_instrcut", "qwen2_5_instruct"]:
        collator = ChatTemplateTestCollator(args, tokenizer=processor)
    else:
        collator = UnifiedTestCollator(args, processor_or_tokenizer=processor)

    collate_fn = _make_indexed_collator(collator) if rollout_file else collator
    test_loader = DataLoader(
        dataset_for_loader,
        batch_size=args.test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=test_sampler,
    )

    if rank == 0:
        print(f"\n{eval_split_lower} 数据集大小 (total): {len(test_data)}")
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

    shard_fp = None
    shard_path = None
    done_path = None
    if rollout_file:
        shard_path = f"{rollout_file}.rank{rank}.jsonl"
        done_path = f"{rollout_file}.rank{rank}.done"
        os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
        # Remove stale markers from previous runs.
        try:
            if done_path and os.path.exists(done_path):
                os.remove(done_path)
        except OSError:
            pass
        shard_fp = open(shard_path, "w")

    with torch.no_grad():
        for prompt_id in prompt_ids:
            if rank == 0:
                print(f"\n评估Prompt {prompt_id}...")

            test_loader.sampler.set_epoch(prompt_id)
            test_loader.dataset.set_prompt(prompt_id)

            iterable = tqdm(
                test_loader, desc=f"Prompt {prompt_id}", disable=(rank != 0)
            )

            for step, batch in enumerate(iterable):
                inputs = batch[0]
                targets = batch[1]
                indices = batch[2] if rollout_file else None

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
                    do_sample=False,
                    top_k=None,
                    top_p=None,
                    logits_processor=logits_processor,
                )

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output_text = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                pred_items = clean_predictions(output_text)

                if shard_fp is not None and indices is not None:
                    scores_list = scores.detach().cpu().tolist()
                    lines = []
                    for i, dataset_index in enumerate(indices):
                        start = i * args.num_beams
                        end = (i + 1) * args.num_beams
                        rec = {
                            "prompt_id": prompt_id,
                            "index": int(dataset_index),
                            "target": targets[i],
                            "predictions": pred_items[start:end],
                            "scores": scores_list[start:end],
                        }
                        lines.append(
                            json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
                        )
                    shard_fp.write("\n".join(lines) + "\n")

    if shard_fp is not None:
        shard_fp.close()

    if rollout_file:
        assert done_path is not None
        with open(done_path, "w") as f:
            f.write("done\n")

        if rank == 0:
            # Wait for all ranks to finish writing shards.
            expected_done = [f"{rollout_file}.rank{r}.done" for r in range(world_size)]
            timeout_s = 24 * 60 * 60
            start = time.time()
            while True:
                if all(os.path.exists(p) for p in expected_done):
                    break
                if time.time() - start > timeout_s:
                    missing = [p for p in expected_done if not os.path.exists(p)]
                    raise TimeoutError(f"Timeout waiting for rollout shards: {missing}")
                time.sleep(1.0)

            num_samples = len(test_data)
            targets = [None] * num_samples
            prompts_cache = {
                str(pid): {
                    "predictions": [None] * num_samples,
                    "scores": [None] * num_samples,
                }
                for pid in prompt_ids
            }

            for r in range(world_size):
                path = f"{rollout_file}.rank{r}.jsonl"
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing rollout shard: {path}")
                with open(path, "r") as fp:
                    for line in fp:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        pid = str(rec["prompt_id"])
                        idx = int(rec["index"])
                        if targets[idx] is None:
                            targets[idx] = rec["target"]
                        prompts_cache[pid]["predictions"][idx] = rec["predictions"]
                        prompts_cache[pid]["scores"][idx] = rec["scores"]

            if any(v is None for v in targets):
                raise RuntimeError("rollout targets 缺失")
            for pid in prompt_ids:
                preds = prompts_cache[str(pid)]["predictions"]
                scs = prompts_cache[str(pid)]["scores"]
                if any(v is None for v in preds):
                    raise RuntimeError(f"rollout predictions 缺失，prompt_id={pid}")
                if any(v is None for v in scs):
                    raise RuntimeError(f"rollout scores 缺失，prompt_id={pid}")

            rollout_payload = {
                "schema_version": 1,
                "meta": {
                    "created_at": int(time.time()),
                    "key": _rollout_key_from_args(args, eval_split_lower),
                },
                "targets": targets,
                "prompts": prompts_cache,
            }
            _atomic_json_dump(rollout_file, rollout_payload)
            print(f"Rollout saved to {rollout_file}")

            metrics = args.metrics.split(",")
            all_items_for_metrics = None
            if args.filter_items:
                all_items_for_metrics = all_items

            all_prompt_results = []
            for prompt_id in prompt_ids:
                pdata = rollout_payload["prompts"][str(prompt_id)]
                predictions = pdata["predictions"]
                scores = pdata["scores"]

                preds_flat = [p for row in predictions for p in row]
                scores_flat = [s for row in scores for s in row]

                topk_res = get_topk_results(
                    preds_flat,
                    scores_flat,
                    rollout_payload["targets"],
                    args.num_beams,
                    all_items=all_items_for_metrics if args.filter_items else None,
                    clean=False,
                )
                metrics_sum = get_metrics_results(topk_res, metrics)
                total = max(len(rollout_payload["targets"]), 1)
                metrics_results = {m: metrics_sum[m] / total for m in metrics_sum}

                all_prompt_results.append(metrics_results)
                print("======================================================")
                print(args.ckpt_path)
                print("======================================================")
                print(f"Prompt {prompt_id} constrained results: ", metrics_results)
                print(f"(Based on {len(rollout_payload['targets'])} total samples)")
                print("======================================================")
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

            print("======================================================")
            print("Mean results: ", mean_results)
            print("Min results: ", min_results)
            print("Max results: ", max_results)
            print("======================================================")

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
                "rollout_cached": False,
            }

            os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
            with open(args.results_file, "w") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            print(f"Results saved to {args.results_file}")

            # Cleanup shard + done files to avoid accumulating large data.
            for r in range(world_size):
                shard = f"{rollout_file}.rank{r}.jsonl"
                marker = f"{rollout_file}.rank{r}.done"
                for p in (shard, marker):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass


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

    try:
        test(args)
    except Exception as exc:
        print(f"Error occurred: {exc}")
        raise
