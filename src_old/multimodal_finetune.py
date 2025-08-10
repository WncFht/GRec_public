import argparse
import os

import torch
import transformers
from collator import MultiModalCollator
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
)
from utils import (
    ensure_dir,
    freeze_original_embeddings_simple,
    load_datasets,
    parse_dataset_args,
    parse_global_args,
    parse_train_args,
    set_seed,
    verify_token_ordering,
)


def setup_environment(args: argparse.Namespace) -> tuple[int, bool]:
    """
    设置训练环境，包括随机种子、目录和分布式训练设置。

    Args:
    ----
        args (argparse.Namespace): 包含配置的参数。

    Returns:
    -------
        tuple[int, bool]: (local_rank, ddp)

    """
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if ddp:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = None

    if local_rank == 0:
        print(f"训练模式: 全量微调 {'DDP' if ddp else '单GPU'}")
        print(f"Device map: {device_map}")
        print(vars(args))
    return local_rank, ddp


def load_and_prepare_model_tokenizer(
    args: argparse.Namespace, local_rank: int
) -> tuple[
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    int,
    ConcatDataset,
    Dataset | None,
]:
    """
    加载基础模型和处理器，并根据数据集准备tokenizer。

    Args:
    ----
        args (argparse.Namespace): 包含配置的参数。
        local_rank (int): 当前进程的rank。

    Returns:
    -------
        tuple: (model, processor, original_vocab_size, train_data, valid_data)

    """
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)

    train_data, valid_data = load_datasets(args)
    new_tokens = train_data.datasets[0].get_new_tokens()
    original_vocab_size = len(processor.tokenizer)

    if local_rank == 0:
        print(f"原始词汇表大小: {original_vocab_size}")
        print(f"需要添加新token数量: {len(new_tokens)}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    add_num = processor.tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))

    verify_token_ordering(processor, original_vocab_size, new_tokens)
    freeze_original_embeddings_simple(model, original_vocab_size)

    new_vocab_size = len(processor.tokenizer)
    model.config.vocab_size = new_vocab_size
    config.vocab_size = new_vocab_size

    if local_rank == 0:
        print(f"添加了 {add_num} 个新token")
        print(f"新词汇表大小: {new_vocab_size}")
        print(f"数据量: {len(train_data)}")
        print(
            f"有效batch size: {args.per_device_batch_size * args.gradient_accumulation_steps * int(os.environ.get('WORLD_SIZE', 1))}"
        )
        print("embedding size: ", model.get_input_embeddings().weight.shape)
        processor.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    return model, processor, original_vocab_size, train_data, valid_data


def get_training_args(args: argparse.Namespace, ddp: bool) -> TrainingArguments:
    """
    构建Hugging Face Trainer的训练参数。

    Args:
    ----
        args (argparse.Namespace): 包含训练配置的参数。
        ddp (bool): 是否使用分布式数据并行。

    Returns:
    -------
        TrainingArguments: Hugging Face的训练参数。

    """
    return TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_step,
        optim=args.optim,
        gradient_checkpointing=True,
        eval_strategy=args.save_and_eval_strategy,
        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        save_total_limit=5,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb",
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
    )


def train(args: argparse.Namespace) -> None:
    """
    主训练函数，协调整个全量微调流程。

    Args:
    ----
        args (argparse.Namespace): 包含所有配置的参数。

    """
    local_rank, ddp = setup_environment(args)

    (
        model,
        processor,
        original_vocab_size,
        train_data,
        valid_data,
    ) = load_and_prepare_model_tokenizer(args, local_rank)

    collator = MultiModalCollator(args, processor)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = get_training_args(args, ddp)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if local_rank == 0:
        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiModalQwenRec Full Finetune"
    )
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    parser.add_argument(
        "--image_path", type=str, default="images", help="Path to images folder"
    )

    args = parser.parse_args()
    train(args)
