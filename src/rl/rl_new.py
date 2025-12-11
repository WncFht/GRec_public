import argparse
import os
from collections import defaultdict

from datasets import Dataset as HFDataset

from trl import GRPOConfig

from ..data_rl import FusionSeqRecDataset, SeqRecDataset
from ..parser import parse_dataset_args, parse_global_args, parse_rl_args
from ..utils import ensure_dir, load_model_for_training, set_seed
from .minionerec_trainer import ReReTrainer
from .reward_fns import (
    format_reward,
    initialize_reward_functions,
    ndcg_rule_reward,
    rule_reward,
)


def debug_prefix_index(tokenizer, base_model_name: str):
    r"""
    辅助函数：打印 '### Response:\\nitem\\n' 的分词结果，方便人工选择 prefix_index。
    不会在训练流程中自动调用，如需查看可以在 main 里手动调用。
    """
    sample_item = "<a_1><b_1><c_1><d_1>"
    text = f"### Response:<|im_end|><|im_start|>assistant\n{sample_item}<|im_end|>"
    tokenized = tokenizer(text)
    ids = tokenized["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"[Debug prefix_index] base_model={base_model_name}")
    print("Text:", repr(text))
    print("IDs :", ids)
    print("Tokens:", tokens)


def main():
    # ====================================================
    # 1. 参数解析 (使用 parser.py)
    # ====================================================
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_rl_args(parser)

    parsed_args = parser.parse_args()  # 扁平对象，传给 utils.* 使用
    num_generations = parsed_args.num_generations

    print(parsed_args)
    # ====================================================
    # 2. 环境设置 (使用 utils.py)
    # ====================================================
    # # 生成 Run Name
    # run_name = make_run_name(parsed_args)
    # parsed_args.run_name = run_name  # 回写到 args 以供 utils 内部使用

    # 设置 WANDB
    # if parsed_args.run_name and parsed_args.run_name != "none":
    #     os.environ["WANDB_PROJECT"] = "rl_rec"
    #     os.environ["WANDB_RUN_NAME"] = run_name
    # else:
    #     os.environ["WANDB_MODE"] = "disabled"

    # 设置随机种子
    set_seed(parsed_args.seed)
    ensure_dir(parsed_args.output_dir)

    # print(f"Run Name: {run_name}")
    print(f"Model Type: {parsed_args.model_type}")
    print(f"Base Model: {parsed_args.base_model}")

    # ====================================================
    # 3. 数据集准备
    # ====================================================
    # 先构造 data_rl 里的 PyTorch Dataset，
    # 再统一转换成 Verl 风格记录并包装成 HF Dataset，
    # 以满足 ReReTrainer 的输入格式要求。

    tasks = parsed_args.tasks.split(",")
    train_datasets = []
    valid_datasets = []
    test_datasets = []

    for task in tasks:
        dataset_list = parsed_args.dataset.split(",")
        for dataset_name in dataset_list:
            train_dataset = None
            valid_dataset = None
            test_dataset = None

            if task.lower() == "seqrec":
                train_dataset = SeqRecDataset(
                    parsed_args,
                    mode="train",
                    dataset=dataset_name,
                )
                if parsed_args.eval_on_valid:
                    valid_dataset = SeqRecDataset(
                        parsed_args,
                        mode="valid",
                        dataset=dataset_name,
                    )
                if parsed_args.eval_on_test:
                    test_dataset = SeqRecDataset(
                        parsed_args,
                        mode="test",
                        dataset=dataset_name,
                    )
            elif task.lower() == "fusionseqrec":
                train_dataset = FusionSeqRecDataset(
                    parsed_args,
                    mode="train",
                    dataset=dataset_name,
                )
                # valid_dataset = FusionSeqRecDataset(
                #     parsed_args,
                #     mode="valid",
                #     dataset=dataset_name,
                # )
                # if parsed_args.eval_on_test:
                #     test_dataset = FusionSeqRecDataset(
                #         parsed_args,
                #         mode="test",
                #         dataset=dataset_name,
                #     )

            if train_dataset is not None:
                train_datasets.append(train_dataset)
                print(
                    f"Task: {task} - dataset: {dataset_name} - train samples: {len(train_dataset)}"
                )
            if test_dataset is not None:
                test_datasets.append(test_dataset)
                print(
                    f"Task: {task} - dataset: {dataset_name} - test samples: {len(test_dataset)}"
                )
            if valid_dataset is not None:
                valid_datasets.append(valid_dataset)
                print(
                    f"Task: {task} - dataset: {dataset_name} - valid samples: {len(valid_dataset)}"
                )

    if not train_datasets:
        msg = "No train datasets constructed. Please check `--tasks` and `--dataset`."
        raise ValueError(msg)

    # ====================================================
    # 4. 模型加载 (使用 utils.load_model_for_training)
    # ====================================================
    # 这个函数封装了: Tokenizer, Resize Embeddings, LoRA, Freeze

    model, processor, orig_vocab, new_vocab, _, embedding_hooks = (
        load_model_for_training(
            args=parsed_args,
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        )
    )

    # 从 processor 获取 tokenizer
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    # 确保 pad_token 存在 (GRPO 必须)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 某些模型可能需要手动设置 pad_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id
    print(f"Using eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"Using pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # 注册 tokenizer 并初始化奖励函数所需的上下文
    if initialize_reward_functions(
        num_generations,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
    ):
        return

    # ====================================================
    # 3.1 转换数据集为 Verl 记录（包含 ground_truth token ids）
    # ====================================================
    print("Processing Train Dataset (to Verl records)...")
    train_records = []
    for ds in train_datasets:
        if hasattr(ds, "to_verl_records"):
            train_records.extend(ds.to_verl_records("train", tokenizer=tokenizer))

    train_dataset = HFDataset.from_list(train_records)
    train_dataset = train_dataset.shuffle(seed=parsed_args.seed)

    test_eval_dataset = None
    if parsed_args.eval_on_test and test_datasets:
        print("Processing Test Dataset (to Verl records)...")
        test_records = []
        for ds in test_datasets:
            if hasattr(ds, "to_verl_records"):
                test_records.extend(ds.to_verl_records("test", tokenizer=tokenizer))
        test_eval_dataset = HFDataset.from_list(test_records) if test_records else None

    valid_eval_dataset = None
    if parsed_args.eval_on_valid and valid_datasets:
        print("Processing Valid Dataset (to Verl records)...")
        valid_records = []
        for ds in valid_datasets:
            if hasattr(ds, "to_verl_records"):
                valid_records.extend(ds.to_verl_records("valid", tokenizer=tokenizer))
        valid_eval_dataset = (
            HFDataset.from_list(valid_records) if valid_records else None
        )

    combined_eval_dataset = test_eval_dataset
    if parsed_args.eval_on_valid:
        if valid_eval_dataset is not None and test_eval_dataset is not None:
            combined_eval_dataset = {
                "test": test_eval_dataset,
                "valid": valid_eval_dataset,
            }
        elif valid_eval_dataset is not None:
            combined_eval_dataset = valid_eval_dataset

    print(f"Train Size: {len(train_dataset)}")
    print(
        f"Test Eval Size: {len(test_eval_dataset) if test_eval_dataset is not None else 0}"
    )
    if parsed_args.eval_on_valid:
        print(
            f"Valid Eval Size: {len(valid_eval_dataset) if valid_eval_dataset is not None else 0}"
        )

    # if True:
    #     debug_prefix_index(tokenizer, "test")
    #     sys.exit()
    # ====================================================
    # 4.1 基于数据集构建 hash_dict（前缀约束）
    # ====================================================
    # 简单的 prefix_index 规则（与原实现保持一致），
    # 如需更精细可以用下方 debug 函数做检查后手动调整。
    base_model_lower = parsed_args.base_model.lower()
    if "llava" in base_model_lower:
        prefix_index = 7
    elif "gpt2" in base_model_lower:
        prefix_index = 4
    else:
        prefix_index = 3

    merged_hash_dict: dict[str, set[int]] = defaultdict(set)
    for ds in train_datasets:
        if hasattr(ds, "build_hash_dict"):
            ds_hash = ds.build_hash_dict(tokenizer, prefix_index=prefix_index)
            for k, vals in ds_hash.items():
                merged_hash_dict[k].update(vals)

    hash_dict = {k: sorted(list(v)) for k, v in merged_hash_dict.items()}
    print(f"Built hash_dict entries: {len(hash_dict)} with prefix_index={prefix_index}")
    # print("10th of the hash_dict")
    # import pprint; pprint.pprint(dict(list(hash_dict.items())[:10]))
    reward_type = parsed_args.reward_type
    if reward_type == "rule":
        reward_fun = [format_reward, rule_reward]
    elif reward_type == "ranking":
        reward_fun = [format_reward, rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = [format_reward, ndcg_rule_reward]

    # ====================================================
    # 6. 配置 Trainer
    # ====================================================
    # 映射参数到 GRPOConfig
    training_args = GRPOConfig(
        output_dir=parsed_args.output_dir,
        save_steps=0.1,
        save_total_limit=20,
        save_only_model=True,
        eval_strategy="steps",
        max_completion_length=parsed_args.max_completion_length,
        num_generations=num_generations,
        temperature=parsed_args.temperature,
        sync_ref_model=parsed_args.sync_ref_model,
        per_device_eval_batch_size=parsed_args.eval_batch_size,
        per_device_train_batch_size=parsed_args.train_batch_size,
        gradient_accumulation_steps=parsed_args.gradient_accumulation_steps,
        eval_steps=parsed_args.eval_step,
        logging_steps=5,
        log_completions=parsed_args.log_completions,
        learning_rate=parsed_args.learning_rate,
        beta=parsed_args.beta,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        num_train_epochs=parsed_args.num_train_epochs,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        report_to="wandb",
    )
    training_args.completion_log_interval = parsed_args.completion_log_interval

    # 初始化自定义 Trainer
    trainer = ReReTrainer(
        model=model,
        base_model=parsed_args.base_model,
        dapo=parsed_args.dapo,
        gspo=parsed_args.gspo,
        noscale=parsed_args.noscale,
        use_prm=parsed_args.use_prm,
        use_sft_loss=parsed_args.use_sft_loss,
        sft_loss_coef=parsed_args.sft_loss_coef,
        add_gt=parsed_args.add_gt,
        dynamic_sampling=parsed_args.dynamic_sampling,
        beam_search=parsed_args.beam_search,
        test_during_training=parsed_args.test_during_training,
        test_beam=parsed_args.test_beam,
        hash_dict=hash_dict,
        prefix_index=prefix_index,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=combined_eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # ====================================================
    # 7. 训练与保存
    # ====================================================
    print("Starting Training...")
    trainer.train()

    print(f"Saving model to {parsed_args.output_dir}")
    trainer.save_model(parsed_args.output_dir)

    if test_eval_dataset is not None:
        print("Running evaluation on test split...")
        test_metrics = trainer.evaluate(
            eval_dataset=test_eval_dataset, metric_key_prefix="test"
        )
        print(f"Test metrics: {test_metrics}")
    if parsed_args.eval_on_valid and valid_eval_dataset is not None:
        print("Running evaluation on valid split...")
        valid_metrics = trainer.evaluate(
            eval_dataset=valid_eval_dataset, metric_key_prefix="valid"
        )
        print(f"Valid metrics: {valid_metrics}")

    # 保存最终 checkpoint
    final_dir = os.path.join(parsed_args.output_dir, "final_checkpoint")
    ensure_dir(final_dir)

    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 保存 token metadata
    if hasattr(model, "config"):
        model.config.save_pretrained(final_dir)

    print("Training Finished.")


if __name__ == "__main__":
    main()
