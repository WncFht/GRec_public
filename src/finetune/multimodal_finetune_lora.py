import os
import sys

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..collator import MultiModalCollator
from ..config import parse_args
from ..type import Args
from ..utils import (
    ensure_dir,
    get_tokenizer,
    load_datasets,
    load_model_for_training,
    set_seed,
)


def setup_environment(args: Args) -> tuple[int, bool]:
    """
    设置训练环境，包括随机种子、目录和分布式训练设置。

    Args:
    ----
        args (Args): 包含配置的参数。

    Returns:
    -------
        tuple[int, bool]: (local_rank, ddp)

    """
    global_args = args.global_args
    set_seed(global_args.seed)
    ensure_dir(global_args.output_dir)

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


def get_collator(
    args: Args, tokenizer_or_processor: AutoProcessor | AutoTokenizer
) -> MultiModalCollator:
    # if args.model_type == "qwen_vl":
    return MultiModalCollator(args, tokenizer_or_processor)
    # else:
    # return Collator(args, tokenizer_or_processor)


def get_train_args(args: Args, ddp: bool) -> TrainingArguments:
    """
    构建Hugging Face Trainer的训练参数。

    Args:
    ----
        args (Args): 包含训练配置的参数。
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
        ddp_find_unused_parameters=False if ddp else None,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="wandb",
        eval_delay=1 if train_args.save_and_eval_strategy == "epoch" else 2000,
    )


def train(args: Args) -> None:
    """
    主训练函数，协调整个LoRA微调流程。

    Args:
    ----
        args (Args): 包含所有配置的参数。

    """
    local_rank, ddp = setup_environment(args)

    # 1. 先加载数据集以获取新词汇
    train_data, valid_data = load_datasets(args)

    # 假设所有子数据集共享相同的词汇扩展
    new_tokens = train_data.datasets[0].get_new_tokens()

    if args.global_args.debug:
        print("len(train_data):", len(train_data))
        print("len(valid_data):", len(valid_data))
        print("new_tokens:", new_tokens)
        # print case of train_data
        print("case of train_data:", train_data.datasets[0].__getitem__(0))

    # 2. 加载模型并传入新词汇进行扩展
    model, tokenizer_or_processor = load_model_for_training(args, new_tokens)

    if local_rank == 0:
        print(
            f"有效batch size: {args.train_args.per_device_batch_size * args.train_args.gradient_accumulation_steps * int(os.environ.get('WORLD_SIZE', 1))}"
        )
        print("embedding size: ", model.get_input_embeddings().weight.shape)
        tokenizer_or_processor.save_pretrained(args.global_args.output_dir)
        config = AutoConfig.from_pretrained(
            args.global_args.base_model, trust_remote_code=True
        )
        config.save_pretrained(args.global_args.output_dir)

    # 3. 准备 Collator 和 Trainer
    collator = get_collator(args, tokenizer_or_processor)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    train_args = get_train_args(args, ddp)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=train_args,
        tokenizer=get_tokenizer(tokenizer_or_processor),
        data_collator=collator,
    )

    model.config.use_cache = False
    if torch.__version__ >= "2" and sys.platform != "win32":
        print("Compiling model...")
        model = torch.compile(model)
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    if local_rank == 0:
        trainer.save_state()
        trainer.save_model(output_dir=args.global_args.output_dir)
        model.save_pretrained(args.global_args.output_dir)
        tokenizer_or_processor.save_pretrained(args.global_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
