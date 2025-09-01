import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle

import torch
import transformers
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
)

from ..collator import MultiModalCollator
from ..parser import parse_dataset_args, parse_global_args, parse_train_args
from ..utils import ensure_dir, load_datasets, load_model_for_training, set_seed


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

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if ddp:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    if local_rank == 0:
        print(f"训练模式: LoRA微调 {'DDP' if ddp else '单GPU'}")
        print(f"Device map: {device_map}")
        print(vars(args))
    return local_rank, ddp


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
        gradient_checkpointing=args.use_gradient_checkpointing,
        eval_strategy=args.save_and_eval_strategy,
        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        # save_total_limit=5,
        load_best_model_at_end=True,
        # deepspeed=train_args.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
    )


def save_new_token_embeddings(
    model: Qwen2VLForConditionalGeneration,
    original_vocab_size: int,
    new_vocab_size: int,
    new_tokens: list[str],
    output_dir: str,
) -> None:
    # 保存新 token 的 embedding 用于后续可视化
    # 对于LoRA模型，需要从base model获取embedding
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
    else:
        base_model = model

    new_token_embeddings = (
        base_model.get_input_embeddings()
        .weight[original_vocab_size:]
        .detach()
        .cpu()
    )

    # 转换为 float32 类型，避免 BFloat16 的兼容性问题
    new_token_embeddings = new_token_embeddings.float()

    # 保存 embedding 和 token 名称
    embedding_info = {
        "embeddings": new_token_embeddings,
        "token_names": new_tokens,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
    }

    embedding_save_path = os.path.join(output_dir, "new_token_embeddings.pkl")
    with open(embedding_save_path, "wb") as f:
        pickle.dump(embedding_info, f)
    print(f"新 token embedding 已保存到: {embedding_save_path}")


def load_and_prepare_model_tokenizer(
    args: argparse.Namespace, local_rank: int
) -> tuple[
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    ConcatDataset,
    Dataset | None,
    list,
    int,
    int,
    list,
]:
    """
    加载基础模型和处理器，并根据数据集准备tokenizer，配置LoRA。

    Args:
    ----
        args (argparse.Namespace): 包含配置的参数。
        local_rank (int): 当前进程的rank。

    Returns:
    -------
        tuple: (model, processor, train_data, valid_data, embedding_hooks,
                original_vocab_size, new_vocab_size, new_tokens)

    """
    # 确保LoRA模式
    args.use_lora = True
    
    # 使用统一的load_model_for_training函数
    model, processor, original_vocab_size, new_vocab_size, new_tokens, embedding_hooks = load_model_for_training(
        args, new_tokens=None, local_rank=local_rank, logger=None
    )

    # 加载数据集
    train_data, valid_data = load_datasets(args)

    if local_rank == 0:
        print(
            f"LoRA配置: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
        )
        print(f"目标模块: {args.lora_target_modules}")
        if hasattr(args, 'lora_modules_to_save') and args.lora_modules_to_save:
            print(f"保存模块: {args.lora_modules_to_save}")
        print(f"添加了 {new_vocab_size - original_vocab_size} 个新token")
        print(f"原始词汇表大小: {original_vocab_size}")
        print(f"新词汇表大小: {new_vocab_size}")
        print(f"数据量: {len(train_data)}")

        # 打印可训练参数统计
        model.print_trainable_parameters()

        if hasattr(args, 'use_gradient_checkpointing') and args.use_gradient_checkpointing:
            effective_batch_size = (
                args.per_device_batch_size
                * args.gradient_accumulation_steps
                * int(os.environ.get("WORLD_SIZE", 1))
            )
        else:
            effective_batch_size = args.per_device_batch_size * int(
                os.environ.get("WORLD_SIZE", 1)
            )
        print(f"有效batch size: {effective_batch_size}")
        print(f"1 epoch steps: {len(train_data) / args.per_device_batch_size}")

        processor.save_pretrained(args.output_dir)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
        config.vocab_size = new_vocab_size
        config.save_pretrained(args.output_dir)

    return (
        model,
        processor,
        train_data,
        valid_data,
        embedding_hooks,
        original_vocab_size,
        new_vocab_size,
        new_tokens,
    )


def train(args: argparse.Namespace) -> None:
    """
    主训练函数，协调整个LoRA微调流程。

    Args:
    ----
        args: 训练参数

    """
    local_rank, ddp = setup_environment(args)

    (
        model,
        processor,
        train_data,
        valid_data,
        embedding_hooks,
        original_vocab_size,
        new_vocab_size,
        new_tokens,
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
        processing_class=processor,
        data_collator=collator,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        print("Compiling model...")
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 清理embedding梯度hook
    if embedding_hooks:
        for hook in embedding_hooks:
            hook.remove()
        print(f"清理了 {len(embedding_hooks)} 个embedding梯度hook")

    save_new_token_embeddings(
        model,
        original_vocab_size,
        new_vocab_size,
        new_tokens,
        args.output_dir,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)
    # 不需要add_lora_args，因为parse_train_args已经包含了所有LoRA参数

    args = parser.parse_args()
    train(args)
