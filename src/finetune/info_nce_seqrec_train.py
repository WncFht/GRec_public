import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import transformers
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
)

from ..collator import MultiModalCollator
from ..info_nce import InfoNCE
from ..parser import (
    parse_dataset_args,
    parse_global_args,
    parse_train_args,
)
from ..utils import (
    ensure_dir,
    freeze_original_embeddings_with_hook,
    load_datasets,
    set_seed,
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
        print(f"训练模式: InfoNCE SeqRec {'DDP' if ddp else '单GPU'}")
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
    training_args = TrainingArguments(
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
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
    )

    # 添加 InfoNCE 相关参数到 training_args 中
    training_args.temperature = getattr(args, "temperature", 0.1)
    training_args.info_nce_weight = getattr(args, "info_nce_weight", 1.0)

    return training_args


def save_new_token_embeddings(
    model: Qwen2VLForConditionalGeneration,
    original_vocab_size: int,
    new_vocab_size: int,
    new_tokens: list[str],
    output_dir: str,
) -> None:
    # 保存新 token 的 embedding 用于后续可视化
    new_token_embeddings = (
        model.get_input_embeddings().weight[original_vocab_size:].detach().cpu()
    )

    # 转换为 float32 类型，避免 BFloat16 的兼容性问题
    new_token_embeddings = new_token_embeddings.float()

    # 保存 embedding 和 token 名称
    import pickle

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
]:
    """
    加载基础模型和处理器，并根据数据集准备tokenizer。

    Args:
    ----
        args_terminal (argparse.Namespace): 包含命令行参数的参数。
        args (argparse.Namespace): 包含配置的参数。
        local_rank (int): 当前进程的rank。

    Returns:
    -------
        tuple: (model, processor, train_data, valid_data)

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
    config.vocab_size = new_vocab_size
    model.config.vocab_size = new_vocab_size

    embedding_hooks = []
    if args.freeze in ["visual", "all"]:
        if hasattr(model, "visual"):
            for name, param in model.visual.named_parameters():
                param.requires_grad = False
            print("冻结视觉模型参数")
        if hasattr(model, "visual") and hasattr(model.visual, "merger"):
            for name, param in model.visual.merger.named_parameters():
                param.requires_grad = False
            print("冻结视觉模型融合层参数")
    if args.freeze in ["embeddings", "all"]:
        embedding_hooks = freeze_original_embeddings_with_hook(
            model, original_vocab_size
        )

    if local_rank == 0:
        print(f"添加了 {add_num} 个新token")
        print(f"新词汇表大小: {new_vocab_size}")
        print(f"数据量: {len(train_data)}")
        if args.use_gradient_checkpointing:
            print(
                f"有效batch size: {args.per_device_batch_size * args.gradient_accumulation_steps * int(os.environ.get('WORLD_SIZE', 1))}"
            )
        else:
            print(
                f"有效batch size: {args.per_device_batch_size * int(os.environ.get('WORLD_SIZE', 1))}"
            )
        print(
            "1 epoch step:",
            len(train_data) / args.per_device_batch_size,
        )
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


class InfoNCETrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        self.info_nce_loss = InfoNCE(
            temperature=kwargs.pop("temperature", 0.1),
            reduction="mean",
            negative_mode="unpaired",
        )
        self.original_vocab_size = kwargs.pop("original_vocab_size", 0)
        self.new_tokens = kwargs.pop("new_tokens", [])
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取输入和标签
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")

        # 标准的模型前向传播
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # 获取标准的交叉熵损失
        ce_loss = outputs.loss

        # 计算 InfoNCE 损失
        info_nce_loss = self.compute_info_nce_loss(model, inputs)

        # 组合损失
        total_loss = ce_loss + self.args.info_nce_weight * info_nce_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def compute_info_nce_loss(self, model, inputs):
        """计算 InfoNCE 损失"""
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        batch_size = input_ids.size(0)

        # 获取模型的输入嵌入层
        embedding_layer = model.get_input_embeddings()

        # 提取新 token 的 embeddings
        new_token_embeddings = embedding_layer.weight[
            self.original_vocab_size :
        ]

        # 找出当前 batch 中涉及的新 token
        unique_new_tokens = set()
        for i in range(batch_size):
            # 只考虑标签中不是 -100 的位置
            valid_labels = labels[i][labels[i] != -100]
            # 找出新 token 的索引
            new_token_indices = (
                valid_labels[valid_labels >= self.original_vocab_size]
                - self.original_vocab_size
            )
            unique_new_tokens.update(new_token_indices.tolist())

        if len(unique_new_tokens) == 0:
            return torch.tensor(0.0, device=input_ids.device)

        unique_new_tokens = list(unique_new_tokens)

        # 为每个新 token 创建查询和正样本
        queries = []
        positive_keys = []
        negative_keys = []

        for token_idx in unique_new_tokens:
            token_embedding = new_token_embeddings[
                token_idx : token_idx + 1
            ]  # [1, dim]

            # 查询和正样本都是当前 token
            queries.append(token_embedding)
            positive_keys.append(token_embedding)

            # 负样本是其他的新 tokens
            negative_indices = [
                i for i in range(len(self.new_tokens)) if i != token_idx
            ]
            if negative_indices:
                negative_embeddings = new_token_embeddings[
                    negative_indices
                ]  # [num_negative, dim]
                negative_keys.append(negative_embeddings)

        if not queries:
            return torch.tensor(0.0, device=input_ids.device)

        # 合并所有查询和键
        queries = torch.cat(queries, dim=0)  # [num_queries, dim]
        positive_keys = torch.cat(positive_keys, dim=0)  # [num_queries, dim]

        if negative_keys:
            negative_keys = torch.cat(
                negative_keys, dim=0
            )  # [total_negative, dim]
        else:
            negative_keys = None

        # 计算 InfoNCE 损失
        info_nce_loss = self.info_nce_loss(
            queries, positive_keys, negative_keys
        )

        return info_nce_loss


def train(args: argparse.Namespace) -> None:
    """
    主训练函数，协调整个 InfoNCE SeqRec 微调流程。

    Args:
    ----
        args

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

    # 创建自定义的 InfoNCE Trainer
    trainer = InfoNCETrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        processing_class=processor,
        data_collator=collator,
        temperature=getattr(args, "temperature", 0.1),
        original_vocab_size=original_vocab_size,
        new_tokens=new_tokens,
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


def add_info_nce_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """添加 InfoNCE 相关的参数"""
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="InfoNCE 损失的温度参数"
    )
    parser.add_argument(
        "--info_nce_weight", type=float, default=1.0, help="InfoNCE 损失的权重"
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)
    parser = add_info_nce_args(parser)

    args = parser.parse_args()
    train(args)
