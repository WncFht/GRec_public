import argparse
import json
import os
from typing import Any

import torch
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
)

from collator import MultiModalCollator
from utils import (
    ensure_dir,
    freeze_original_embeddings_with_hook,
    load_datasets,
    parse_dataset_args,
    parse_global_args,
    parse_train_args,
    set_seed,
    verify_token_ordering,
)


def save_model_and_tokens(
    model: PeftModel,
    processor: AutoProcessor,
    output_dir: str,
    original_vocab_size: int,
) -> None:
    """
    保存LoRA模型、处理器和新增的token embeddings。

    此函数负责将训练好的LoRA适配器、处理器、新增token的embedding以及
    相关的元信息保存到指定目录。

    Args:
    ----
        model (PeftModel): 训练好的 PEFT 模型。
        processor (AutoProcessor): 包含更新后 tokenizer 的处理器。
        output_dir (str): 保存模型和文件的输出目录。
        original_vocab_size (int): 原始模型的词汇表大小。

    """
    # # 1. 保存LoRA适配器
    # model.save_pretrained(output_dir)

    # # 2. 保存processor
    # processor.save_pretrained(output_dir)

    # # 3. 提取并保存新增token的embeddings (包括 input_embeddings 和 lm_head)
    # new_token_embeddings = {}
    # for name, param in model.named_parameters():
    #     # 寻找被标记为需要梯度的、词汇表大小相关的参数
    #     if (
    #         param.requires_grad
    #         and len(param.shape) > 1
    #         and param.shape[0] == len(processor.tokenizer)
    #     ):
    #         if "embed_tokens" in name:
    #             key = "input_embeddings"
    #             new_token_embeddings[key] = (
    #                 param[original_vocab_size:].detach().cpu()
    #             )
    #             print(
    #                 f"提取新增 {key} embeddings: "
    #                 f"{new_token_embeddings[key].shape}"
    #             )
    #         elif "lm_head" in name:
    #             key = "lm_head"
    #             new_token_embeddings[key] = (
    #                 param[original_vocab_size:].detach().cpu()
    #             )
    #             print(
    #                 f"提取新增 {key} embeddings: "
    #                 f"{new_token_embeddings[key].shape}"
    #             )

    # if new_token_embeddings:
    #     torch.save(
    #         new_token_embeddings,
    #         os.path.join(output_dir, "new_token_embeddings.pt"),
    #     )

    # 4. 保存元信息
    meta_info = {
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": len(processor.tokenizer),
        "num_new_tokens": len(processor.tokenizer) - original_vocab_size,
        "frozen_original_tokens": True,
    }
    with open(os.path.join(output_dir, "token_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"模型保存完成到: {output_dir}")
    # print("- LoRA适配器: adapter_model.safetensors")
    # print(
    #     "- 新增token embeddings: new_token_embeddings.pt (可能包含input_embeddings和lm_head)"
    # )
    # print("- Tokenizer: tokenizer相关文件")
    print("- 元信息: token_meta.json")


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
        print(f"训练模式: {'DDP' if ddp else '单GPU'}")
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

    new_vocab_size = len(processor.tokenizer)
    model.config.vocab_size = new_vocab_size
    config.vocab_size = new_vocab_size

    if local_rank == 0:
        print(f"添加了 {add_num} 个新token")
        print(f"新词汇表大小: {new_vocab_size}")
        processor.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    return model, processor, original_vocab_size, train_data, valid_data


def configure_lora_model(
    model: Qwen2_5_VLForConditionalGeneration,
    args: argparse.Namespace,
    original_vocab_size: int,
) -> tuple[PeftModel, list[Any]]:
    """
    配置并应用LoRA到模型。

    Args:
    ----
        model (Qwen2_5_VLForConditionalGeneration): 基础模型。
        args (argparse.Namespace): 包含LoRA配置的参数。
        original_vocab_size (int): 原始词汇表大小。

    Returns:
    -------
        tuple[PeftModel, list[Any]]: (lora_model, embedding_hooks)

    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    lora_model = get_peft_model(model, lora_config)
    embedding_hooks = freeze_original_embeddings_with_hook(
        lora_model, original_vocab_size
    )
    return lora_model, embedding_hooks


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
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="wandb",
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
    )


def train(args: argparse.Namespace) -> None:
    """
    主训练函数，协调整个LoRA微调流程。

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

    lora_model, embedding_hooks = configure_lora_model(
        model, args, original_vocab_size
    )

    collator = MultiModalCollator(args, processor)

    if not ddp and torch.cuda.device_count() > 1:
        lora_model.is_parallelizable = True
        lora_model.model_parallel = True

    training_args = get_training_args(args, ddp)

    trainer = transformers.Trainer(
        model=lora_model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )

    lora_model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if local_rank == 0:
        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)

        for hook in embedding_hooks:
            hook.remove()

        save_model_and_tokens(
            lora_model, processor, args.output_dir, original_vocab_size
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiModalQwenRec with LoRA")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    # LoRA相关参数：是否在modules_to_save中保存embedding模块
    parser.add_argument(
        "--save_embedding_modules",
        action="store_true",
        help="Whether to save embedding modules in LoRA modules_to_save",
    )

    # 物品图片相关参数：图片路径
    parser.add_argument("--image_path", type=str, default="images")
    # parser.add_argument('--alignment_data_ratio', type=float, default=0.3)

    args = parser.parse_args()
    train(args)
