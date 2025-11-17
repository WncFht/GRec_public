import argparse
import os
import sys
import math
import random
import numpy as np
import torch
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(
    str(root)
)  # add GRec/src to path so relative modules can be imported

from parser import parse_global_args, parse_dataset_args, parse_train_args
from data_rl import (
    SeqRecDataset,
    FusionSeqRecDataset,
    ItemFeatDataset,
    dataset_to_text_samples,
    samples_to_hf_dataset,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def build_args():
    parser = argparse.ArgumentParser()

    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)
    parser.add_argument(
        "--task",
        type=str,
        default="seqrec",
        help="which task to run: seqrec|fusionseqrec|item2index",
    )
    return parser.parse_args()


def main():
    args = build_args()

    # build train and eval dataset objects using dataset mode (do not re-split)
    task = args.task.lower()
    if task == "seqrec":
        ds_train = SeqRecDataset(args, args.dataset, mode="train")
        ds_eval = SeqRecDataset(args, args.dataset, mode="valid")
    elif task == "fusionseqrec":
        ds_train = FusionSeqRecDataset(args, args.dataset, mode="train")
        ds_eval = FusionSeqRecDataset(args, args.dataset, mode="valid")
    elif task in ("item2index", "item"):
        ds_train = ItemFeatDataset(
            args, args.dataset, task="item2index", mode="train"
        )
        ds_eval = ItemFeatDataset(
            args, args.dataset, task="item2index", mode="valid"
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    print(f"Preparing samples from dataset: {task}")
    samples_train = dataset_to_text_samples(ds_train, mode="train")
    samples_eval = dataset_to_text_samples(ds_eval, mode="valid")
    print(
        f"Collected train: {len(samples_train)} samples, eval: {len(samples_eval)} samples"
    )

    # convert to HuggingFace Dataset
    try:
        train_hf = samples_to_hf_dataset(samples_train)
        eval_hf = samples_to_hf_dataset(samples_eval)
    except Exception as e:
        print("Failed to create HF dataset:", e)
        return

    print("Train HF dataset size:", len(train_hf))

    # load model/tokenizer
    model_name = args.base_model
    print("Loading model and tokenizer:", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # build prompt2history and history2target mappings from records/samples
    prompt2history = {}
    history2target = {}

    # prefer dataset-provided mappings if present
    for d in (ds_train, ds_eval):
        if hasattr(d, "prompt2history"):
            prompt2history.update(getattr(d, "prompt2history"))
        if hasattr(d, "history2target"):
            history2target.update(getattr(d, "history2target"))

    # fallback: build mappings from samples collected (extra_info)
    for s in samples_train + samples_eval:
        prompt = s.get("prompt", "")
        completion = s.get("completion", "")
        extra = s.get("extra_info", {})
        history = (
            extra.get("inters")
            or extra.get("history")
            or extra.get("history_str")
            or ""
        )
        if prompt and history:
            prompt2history.setdefault(prompt, history)
            history2target.setdefault(history, completion)

    # prepare ndcg weights
    num_generations = getattr(
        args, "num_generations", getattr(args, "num_generations", 16)
    )
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    ndcg_rewards = [-elm / sum(ndcg_rewards) for elm in ndcg_rewards]

    def ndcg_rule_reward(prompts, completions):
        history = [prompt2history.get(p, "") for p in prompts]
        targets = [history2target.get(h, "") for h in history]
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):
            if completion.strip('\n"') == targets[i].strip('\n"'):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i % num_generations])

            if (i + 1) % num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []

        return rewards

    def rule_reward(prompts, completions):
        history = [prompt2history.get(p, "") for p in prompts]
        targets = [history2target.get(h, "") for h in history]
        rewards = []

        for i, completion in enumerate(completions):
            if completion.strip('\n" ') == targets[i].strip('\n" '):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    # Try to run GRPO training if trl is available
    try:
        from trl import GRPOConfig, GRPOTrainer

        # map args to GRPOConfig parameters (follow MiniOneRec/rl.py)
        num_train_epochs = getattr(
            args, "num_train_epochs", getattr(args, "epochs", 1)
        )
        train_batch_size = getattr(
            args, "train_batch_size", getattr(args, "per_device_batch_size", 32)
        )
        eval_batch_size = getattr(args, "eval_batch_size", train_batch_size)

        training_args = GRPOConfig(
            output_dir=args.output_dir,
            save_steps=0.1,
            save_total_limit=20,
            eval_strategy="steps",
            max_completion_length=128,
            num_generations=getattr(args, "num_generations", 16),
            temperature=getattr(args, "temperature", 1.0),
            sync_ref_model=getattr(args, "sync_ref_model", False),
            per_device_eval_batch_size=eval_batch_size,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=getattr(
                args, "gradient_accumulation_steps", 1
            ),
            eval_steps=getattr(args, "eval_step", 0.199),
            logging_steps=1,
            learning_rate=getattr(args, "learning_rate", 1e-6),
            beta=getattr(args, "beta", 0.04),
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            num_train_epochs=num_train_epochs,
            bf16=True,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            save_strategy="steps",
            report_to=(
                args.wandb_project
                if getattr(args, "wandb_project", "")
                else "none"
            ),
            run_name=getattr(args, "wandb_run_name", None),
        )

        print("Starting GRPOTrainer (trl) training...")
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=eval_hf,
        )
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    except Exception as e:
        print("trl GRPOTrainer not available or failed to start training:", e)
        # fallback: save datasets to disk so external trainer can use them
        out_dir = os.path.join(args.output_dir, "hf_data")
        os.makedirs(out_dir, exist_ok=True)
        train_out = os.path.join(out_dir, "train")
        eval_out = os.path.join(out_dir, "eval")
        os.makedirs(train_out, exist_ok=True)
        os.makedirs(eval_out, exist_ok=True)
        try:
            train_hf.save_to_disk(train_out)
            eval_hf.save_to_disk(eval_out)
            print(
                f"Saved HF datasets to {out_dir}. You can use them with your RL trainer."
            )
        except Exception as ex:
            print("Failed to save HF datasets:", ex)


if __name__ == "__main__":
    main()
