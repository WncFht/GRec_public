import argparse
import os

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from collator import Collator
from utils import *


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if ddp:
        device_map = {"": local_rank}
        torch.cuda.set_device(local_rank)
    else:
        device_map = None

    if local_rank == 0:
        print(f"训练模式: {'DDP' if ddp else '单GPU'}")
        print(f"Device map: {device_map}")
        print(vars(args))

    # 1. 先加载原始config和tokenizer
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )

    # Qwen tokenizer的pad_token设置
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0

    gradient_checkpointing = True

    # 2. 加载数据并获取新token
    train_data, valid_data = load_datasets(args)
    new_tokens = train_data.datasets[0].get_new_tokens()

    if local_rank == 0:
        print(f"需要添加 {len(new_tokens)} 个新token")
        print(f"原始词汇表大小: {len(tokenizer)}")

    # 3. 先用原始config加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,  # 使用原始config
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    if local_rank == 0:
        print(
            f"模型加载完成，embedding层大小: {model.get_input_embeddings().weight.shape}"
        )

    # 4. 添加新token并调整模型
    add_num = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 5. 更新config并保存
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"添加了 {add_num} 个新token")
        print(f"新词汇表大小: {len(tokenizer)}")
        print(
            f"调整后embedding层大小: {model.get_input_embeddings().weight.shape}"
        )
        print(f"数据量: {len(train_data)}")
        print(
            f"有效batch size: {args.per_device_batch_size * args.gradient_accumulation_steps * world_size}"
        )

        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = Collator(args, tokenizer)

    # 其余代码保持不变...
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
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
            gradient_checkpointing=gradient_checkpointing,
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
            report_to=None,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if local_rank == 0:
        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3Rec")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
