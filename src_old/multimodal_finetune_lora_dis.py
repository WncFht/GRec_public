import argparse
import json
import os

import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
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


def setup_distributed():
    """初始化分布式训练环境"""
    # 检查环境变量中是否存在RANK和WORLD_SIZE，这些变量由torch.distributed.launch或类似的启动器设置
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])  # 当前进程的全局rank
        world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
        local_rank = int(os.environ["LOCAL_RANK"])  # 当前进程在当前节点上的rank
    else:
        # 如果没有设置分布式环境变量，则表示不使用分布式模式
        print("Not using distributed mode")
        return (
            False,
            0,
            1,
            0,
        )  # 返回非分布式标志，rank为0，world_size为1，local_rank为0

    # 设置当前CUDA设备
    torch.cuda.set_device(local_rank)
    # 初始化进程组，使用NCCL作为后端，并指定初始化方法为环境变量
    dist.init_process_group(backend="nccl", init_method="env://")
    # 等待所有进程都到达这里，确保同步
    dist.barrier()

    # 返回分布式标志、全局rank、总进程数和本地rank
    return True, rank, world_size, local_rank


def save_model_with_new_tokens(
    model, processor, output_dir, original_vocab_size, is_main_process
):
    """只在主进程保存模型，包括LoRA适配器、处理器和新增的token embeddings"""
    # 只有主进程才执行保存操作
    if not is_main_process:
        return

    # 保存LoRA适配器（包含微调后的权重）
    model.save_pretrained(output_dir)
    # 保存处理器（包含tokenizer）
    processor.save_pretrained(output_dir)

    # 提取并保存新增token的embeddings
    new_token_embeddings = {}
    # 遍历模型的命名参数，查找词嵌入层
    for name, param in model.named_parameters():
        # 如果参数名中包含"embed_tokens"且该参数是可训练的（requires_grad=True）
        if "embed_tokens" in name and param.requires_grad:
            # 如果embedding层的尺寸大于原始词汇表大小，说明其中包含新增token的embedding
            if param.shape[0] > original_vocab_size:
                # 提取新增token部分的embedding，并将其从GPU转移到CPU，解除计算图连接
                new_token_embeddings["input_embeddings"] = (
                    param[original_vocab_size:].detach().cpu()
                )
                print(
                    f"提取新增input embeddings: {new_token_embeddings['input_embeddings'].shape}"
                )

    # 如果成功提取到新增token的embeddings，则保存到文件
    if new_token_embeddings:
        torch.save(
            new_token_embeddings,
            os.path.join(output_dir, "new_token_embeddings.pt"),
        )

    # 保存元信息，记录词汇表大小和新增token数量等
    meta_info = {
        "original_vocab_size": original_vocab_size,  # 原始词汇表大小
        "new_vocab_size": len(processor.tokenizer),  # 新的词汇表大小
        "num_new_tokens": len(processor.tokenizer)
        - original_vocab_size,  # 新增token数量
        "frozen_original_tokens": True,  # 标记原始token是否被冻结
    }
    # 将元信息保存为JSON文件
    with open(os.path.join(output_dir, "token_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"模型保存完成到: {output_dir}")


def train(args):
    # 初始化分布式环境并获取相关信息
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0  # 判断当前是否为主进程（rank 0）

    # 只有主进程才进行目录创建和打印训练配置信息
    if is_main_process:
        print(f"分布式训练: {is_distributed}, 世界大小: {world_size}")
        print(f"本地rank: {local_rank}, 全局rank: {rank}")
        ensure_dir(args.output_dir)  # 确保输出目录存在

    set_seed(args.seed)  # 设置随机种子，确保可复现性

    # 加载预训练模型和处理器
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)

    # 加载训练和验证数据集
    train_data, valid_data = load_datasets(args)
    # 获取数据集中需要添加的新tokens
    new_tokens = train_data.datasets[0].get_new_tokens()
    # 记录原始tokenizer的词汇表大小
    original_vocab_size = len(processor.tokenizer)

    # 只有主进程才打印词汇表相关信息
    if is_main_process:
        print(f"原始词汇表大小: {original_vocab_size}")
        print(f"需要添加新token数量: {len(new_tokens)}")

    # 加载Qwen2.5_VLForConditionalGeneration模型
    # 使用low_cpu_mem_usage=True减少CPU内存占用
    # device_map=None让Trainer和DDP来管理设备分配
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
        if args.bf16
        else torch.float16,  # 根据参数选择数据类型
        device_map=None,  # 不使用device_map，让DDP处理
        low_cpu_mem_usage=True,  # 减少CPU内存使用
    )

    # 向tokenizer中添加新的tokens，并调整模型词嵌入层的大小
    add_num = processor.tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))

    # 只有主进程才验证新token的顺序
    if is_main_process:
        verify_token_ordering(processor, original_vocab_size, new_tokens)

    # 更新模型配置中的词汇表大小
    new_vocab_size = len(processor.tokenizer)
    model.config.vocab_size = new_vocab_size
    config.vocab_size = new_vocab_size

    # 只有主进程才保存processor和config
    if is_main_process:
        print(f"添加了 {add_num} 个新token")
        print(f"新词汇表大小: {new_vocab_size}")
        processor.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    # 配置LoRA（Low-Rank Adaptation）微调
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
        inference_mode=False,  # 训练模式
        r=args.lora_r,  # LoRA的秩
        lora_alpha=args.lora_alpha,  # LoRA的缩放因子
        lora_dropout=args.lora_dropout,  # LoRA的dropout率
        target_modules=args.lora_target_modules.split(
            ","
        ),  # 应用LoRA的目标模块，通过逗号分隔字符串解析
        bias="none",  # 不对bias进行微调
        modules_to_save=[
            "embed_tokens",
            "lm_head",
        ],  # 需要保存的模块，这里包括词嵌入层和语言模型头
    )

    # 将模型转换为LoRA模型
    model = get_peft_model(model, lora_config)

    # 选择性冻结原始词嵌入，通过hook实现，确保只更新新token的embedding
    embedding_hooks = freeze_original_embeddings_with_hook(
        model, original_vocab_size
    )

    # 将模型移动到当前GPU设备
    model = model.to(f"cuda:{local_rank}")

    # 数据整理器，用于将数据集中的样本整理成批次
    collator = MultiModalCollator(args, processor)

    # 配置训练参数
    training_args = transformers.TrainingArguments(
        seed=args.seed,  # 随机种子
        per_device_train_batch_size=args.per_device_batch_size,  # 每个设备的训练批次大小
        per_device_eval_batch_size=args.per_device_batch_size,  # 每个设备的评估批次大小
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积步数
        warmup_ratio=args.warmup_ratio,  # 学习率预热比例
        num_train_epochs=args.epochs,  # 训练总epoch数
        learning_rate=args.learning_rate,  # 学习率
        weight_decay=args.weight_decay,  # 权重衰减
        lr_scheduler_type=args.lr_scheduler_type,  # 学习率调度器类型
        fp16=args.fp16,  # 是否使用FP16精度
        bf16=args.bf16,  # 是否使用BF16精度
        logging_steps=args.logging_step,  # 日志记录步数
        optim=args.optim,  # 优化器类型
        gradient_checkpointing=args.gradient_checkpointing,  # 是否使用梯度检查点以节省内存
        eval_strategy=args.save_and_eval_strategy,  # 评估策略（例如：按epoch或按步数）
        save_strategy=args.save_and_eval_strategy,  # 保存策略
        eval_steps=args.save_and_eval_steps,  # 评估间隔步数
        save_steps=args.save_and_eval_steps,  # 保存检查点间隔步数
        output_dir=args.output_dir,  # 输出目录
        save_total_limit=args.epochs + 1,  # 最多保存的检查点数量
        load_best_model_at_end=True,  # 训练结束时是否加载最佳模型
        ddp_find_unused_parameters=False,  # 分布式训练中是否查找未使用的参数
        dataloader_num_workers=args.dataloader_num_workers,  # 数据加载器工作进程数
        remove_unused_columns=False,  # 是否移除数据集中的未使用列
        report_to=["none"],  # 不向任何服务报告训练结果
        eval_delay=1
        if args.save_and_eval_strategy == "epoch"
        else 2000,  # 评估延迟
        # 显存优化选项
        dataloader_pin_memory=True,  # 是否将数据加载到CUDA固定内存
        # save_only_model=True,  # 只保存模型权重, DeepSpeed不兼容 (注释掉)
        # DeepSpeed配置
        deepspeed=args.deepspeed_config_file
        if args.use_deepspeed
        else None,  # DeepSpeed配置文件路径
        # 分布式设置
        local_rank=local_rank,  # 当前进程的本地rank
    )

    # 创建Hugging Face Trainer实例
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )

    # 禁用模型缓存，有助于节省内存（特别是在训练阶段）
    model.config.use_cache = False

    # 开始训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 只在主进程中执行保存操作
    if is_main_process:
        trainer.save_state()  # 保存训练器的状态（例如优化器、调度器状态）
        trainer.save_model(output_dir=args.output_dir)  # 保存最终的模型权重

        # 清理之前添加的embedding hooks，避免内存泄漏
        for hook in embedding_hooks:
            hook.remove()

        # 保存LoRA模型和新增的token embeddings
        save_model_with_new_tokens(
            model,
            processor,
            args.output_dir,
            original_vocab_size,
            is_main_process,
        )


if __name__ == "__main__":
    # 创建ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="MultiModalQwenRec with LoRA - Distributed"  # 命令行工具的描述
    )
    # 添加全局参数
    parser = parse_global_args(parser)
    # 添加训练参数
    parser = parse_train_args(parser)
    # 添加数据集参数
    parser = parse_dataset_args(parser)

    # LoRA相关参数：是否在modules_to_save中保存embedding模块
    parser.add_argument(
        "--save_embedding_modules",
        action="store_true",  # 如果命令行中存在此参数，则为True
        help="Whether to save embedding modules in LoRA modules_to_save",
    )

    # 物品图片相关参数：图片路径
    parser.add_argument("--image_path", type=str, default="images")

    # 显存优化参数：是否使用梯度检查点
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory",
    )
    # 显存优化参数：数据加载器工作进程数
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="Number of workers for dataloader",
    )

    # DeepSpeed参数：是否使用DeepSpeed
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training",
    )
    # DeepSpeed参数：DeepSpeed配置文件路径
    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default="deepspeed_config.json",
        help="DeepSpeed configuration file",
    )

    # 解析所有命令行参数
    args = parser.parse_args()
    # 调用训练函数，开始训练过程
    train(args)
