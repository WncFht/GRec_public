import os
import sys

import torch
import transformers
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
)

from ..collator import Collator
from ..config import parse_args
from ..type import Args
from ..utils import (
    ensure_dir,
    load_datasets,
    set_seed,
    verify_token_ordering,
)


def setup_environment(args: Args) -> tuple[int, bool]:
    """
    设置训练环境，包括随机种子、目录和分布式训练设置。

    Args:
    ----
        args (argparse.Namespace): 包含配置的参数。

    Returns:
    -------
        tuple[int, bool]: (local_rank, ddp)

    """
    global_args = args.global_args
    set_seed(global_args.seed)
    ensure_dir(global_args.output_dir)

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
        print(f"训练模式: 全量微调 {'DDP' if ddp else '单GPU'}")
        print(f"Device map: {device_map}")
        print(vars(args))
    return local_rank, ddp


def get_tokenizer(
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
) -> AutoTokenizer:
    """从 Processor 或 Tokenizer 中获取 Tokenizer"""
    if isinstance(tokenizer_or_processor, AutoProcessor):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def load_and_prepare_model_tokenizer(
    args: Args, local_rank: int
) -> tuple[
    Qwen2_5_VLForConditionalGeneration
    | LlamaForCausalLM
    | AutoModelForCausalLM,
    AutoProcessor | AutoTokenizer,
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
    config = AutoConfig.from_pretrained(
        args.global_args.base_model, trust_remote_code=True
    )
    if args.global_args.model_type == "qwen_vl":
        tokenizer_or_processor = AutoProcessor.from_pretrained(
            args.global_args.base_model, trust_remote_code=True
        )
    elif args.global_args.model_type == "llama":
        tokenizer_or_processor = LlamaTokenizer.from_pretrained(
            args.global_args.base_model,
            model_max_length=args.train_args.model_max_length,
            padding_side="right",
            trust_remote_code=True,
        )
        tokenizer_or_processor.pad_token_id = 0
    elif args.global_args.model_type == "qwen":
        tokenizer_or_processor = AutoTokenizer.from_pretrained(
            args.global_args.base_model
        )
    else:
        raise ValueError(
            f"Unsupported model_type: {args.global_args.model_type}"
        )

    train_data, valid_data = load_datasets(args)
    new_tokens = train_data.datasets[0].get_new_tokens()
    if args.global_args.model_type == "qwen_vl":
        tokenizer = tokenizer_or_processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor
    original_vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"原始词汇表大小: {original_vocab_size}")
        print(f"需要添加新token数量: {len(new_tokens)}")

    if args.global_args.model_type == "qwen_vl":
        model_class = Qwen2_5_VLForConditionalGeneration
    elif args.global_args.model_type in ["llama", "qwen"]:
        model_class = AutoModelForCausalLM
    else:
        raise ValueError(
            f"Unsupported model_type: {args.global_args.model_type}"
        )

    model = model_class.from_pretrained(
        args.global_args.base_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.train_args.bf16 else None,
    )

    add_num = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    verify_token_ordering(tokenizer, original_vocab_size, new_tokens)

    new_vocab_size = len(tokenizer)
    config.vocab_size = new_vocab_size
    model.config.vocab_size = new_vocab_size

    if local_rank == 0:
        print(f"添加了 {add_num} 个新token")
        print(f"新词汇表大小: {new_vocab_size}")
        print(f"数据量: {len(train_data)}")
        print(
            f"有效batch size: {args.train_args.per_device_batch_size * args.train_args.gradient_accumulation_steps * int(os.environ.get('WORLD_SIZE', 1))}"
        )
        print("embedding size: ", model.get_input_embeddings().weight.shape)
        if args.global_args.model_type == "qwen_vl":
            tokenizer_or_processor.save_pretrained(args.global_args.output_dir)
        tokenizer.save_pretrained(args.global_args.output_dir)
        config.save_pretrained(args.global_args.output_dir)
    print("=" * 50)
    print("model_type:", args.global_args.model_type)
    for name, param in model.named_parameters():
        print(
            f"{name}: {tuple(param.shape)}, reqires_grad={name, param.requires_grad}"
        )
    if args.global_args.model_type == "qwen_vl":
        if hasattr(model, "visual"):
            for name, param in model.visual.named_parameters():
                param.requires_grad = False
            print("冻结视觉模型参数")
        if hasattr(model, "visual") and hasattr(model.visual, "merger"):
            for name, param in model.visual.merger.named_parameters():
                param.requires_grad = False
            print("冻结视觉模型融合层参数")
    return (
        model,
        tokenizer_or_processor,
        original_vocab_size,
        train_data,
        valid_data,
    )


def get_training_args(args: Args, ddp: bool) -> TrainingArguments:
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
    global_args = args.global_args
    train_args = args.train_args
    return TrainingArguments(
        seed=global_args.seed,
        per_device_train_batch_size=train_args.per_device_batch_size,
        per_device_eval_batch_size=train_args.per_device_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        warmup_ratio=train_args.warmup_ratio,
        num_train_epochs=train_args.epochs,
        learning_rate=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
        lr_scheduler_type=train_args.lr_scheduler_type,
        fp16=train_args.fp16,
        bf16=train_args.bf16,
        logging_steps=train_args.logging_step,
        optim=train_args.optim,
        gradient_checkpointing=True,
        eval_strategy=train_args.save_and_eval_strategy,
        save_strategy=train_args.save_and_eval_strategy,
        eval_steps=train_args.save_and_eval_steps,
        save_steps=train_args.save_and_eval_steps,
        output_dir=global_args.output_dir,
        # save_total_limit=5,
        load_best_model_at_end=True,
        deepspeed=train_args.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="wandb",
        eval_delay=1 if train_args.save_and_eval_strategy == "epoch" else 2000,
    )


def train(args: Args) -> None:
    """
    主训练函数，协调整个全量微调流程。

    Args:
    ----
        args (argparse.Namespace): 包含所有配置的参数。

    """
    local_rank, ddp = setup_environment(args)

    (
        model,
        tokenizer_or_processor,
        original_vocab_size,
        train_data,
        valid_data,
    ) = load_and_prepare_model_tokenizer(args, local_rank)

    collator = Collator(args, tokenizer_or_processor)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = get_training_args(args, ddp)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=get_tokenizer(tokenizer_or_processor),
        data_collator=collator,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        print("Compiling model...")
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=args.train_args.resume_from_checkpoint)

    trainer.save_state()
    trainer.save_model(output_dir=args.global_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
