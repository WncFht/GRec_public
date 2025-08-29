import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
)

from ..collator import MultiModalCollator
from ..parser import parse_dataset_args, parse_global_args, parse_train_args
from ..utils import ensure_dir, load_datasets, set_seed


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
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=True,
    )
    if args.model_type == "qwen2_vl":
        model_class = Qwen2VLForConditionalGeneration
    elif args.model_type == "qwen2_5_vl":
        model_class = Qwen2_5_VLForConditionalGeneration
    elif args.model_type == "llava_onevision":
        model_class = LlavaOnevisionForConditionalGeneration

    model = model_class.from_pretrained(
        args.base_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    train_data, valid_data = load_datasets(args)
    new_tokens = train_data.datasets[0].get_new_tokens()

    tokenizer = processor.tokenizer

    original_vocab_size = len(tokenizer)
    add_num = tokenizer.add_tokens(new_tokens)
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    if args.model_type != "llava_onevision":
        config.vocab_size = new_vocab_size
        model.config.vocab_size = new_vocab_size

    # 配置LoRA - 使用parser.py中已定义的参数
    # 解析target_modules和modules_to_save（它们在args中是逗号分隔的字符串）
    target_modules = args.lora_target_modules.split(",")
    modules_to_save = (
        args.lora_modules_to_save.split(",")
        if args.lora_modules_to_save
        else []
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=modules_to_save,  # 默认包含embed_tokens和lm_head
    )

    model = get_peft_model(model, lora_config)

    # 如果有checkpoint，加载LoRA权重
    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.safetensors"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.bin"
            )

        if os.path.exists(checkpoint_name):
            if local_rank == 0:
                print(f"从检查点加载LoRA权重: {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            model = set_peft_model_state_dict(model, adapters_weights)
        elif local_rank == 0:
            print(f"未找到检查点: {checkpoint_name}")

    # 根据args.freeze参数决定是否冻结原始token的embeddings
    embedding_hooks = []

    if args.freeze in ["embeddings", "all"]:
        # 冻结原始token的embeddings
        base_model = model.get_base_model()
        if hasattr(base_model, "get_input_embeddings"):
            embed_layer = base_model.get_input_embeddings()

            def selective_embedding_hook(grad):
                """选择性地将原始token的梯度置零"""
                if grad is not None:
                    new_grad = grad.clone()
                    new_grad[:original_vocab_size] = 0.0
                    return new_grad
                return grad

            if embed_layer.weight.requires_grad:
                hook = embed_layer.weight.register_hook(
                    selective_embedding_hook
                )
                embedding_hooks.append(hook)
                if local_rank == 0:
                    print(
                        f"冻结原始词汇表embeddings（前{original_vocab_size}个token）"
                    )

    # 处理视觉模型冻结
    if args.freeze in ["visual", "all"]:
        if hasattr(model, "visual"):
            for name, param in model.visual.named_parameters():
                param.requires_grad = False
            print("冻结视觉模型参数")
        if hasattr(model, "visual") and hasattr(model.visual, "merger"):
            for name, param in model.visual.merger.named_parameters():
                param.requires_grad = False
            print("冻结视觉模型融合层参数")

    if local_rank == 0:
        print(
            f"LoRA配置: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
        )
        print(f"目标模块: {target_modules}")
        print(f"保存模块: {modules_to_save}")
        print(f"添加了 {add_num} 个新token")
        print(f"原始词汇表大小: {original_vocab_size}")
        print(f"新词汇表大小: {new_vocab_size}")
        print(f"数据量: {len(train_data)}")

        # 打印可训练参数统计
        model.print_trainable_parameters()

        if args.use_gradient_checkpointing:
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
