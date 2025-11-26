import argparse
import datetime
import json
import os
import random
import re
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import ConcatDataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    InstructBlipForConditionalGeneration,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    T5ForConditionalGeneration,
)

from src.data import (
    FusionSeqRecDataset,
    ItemFeatDataset,
    MultimodalDataset,
    MultimodalSeqRecDataset,
    SeqRecDataset,
    SeqRectWithoutItemIDDataset_1,
    SeqRecWithTitleDataset,
    TextEnrichDataset,
    TextEnrichWihtoutItemIDDataset,
)

# ----------------- 模型和分词器加载工具 -----------------

MODEL_CONFIG = {
    "qwen2_5_vl": {
        "model_class": Qwen2_5_VLForConditionalGeneration,
        "processor_class": AutoProcessor,
        "from_pretrained_kwargs": {"trust_remote_code": True},
    },
    "qwen2_vl": {
        "model_class": Qwen2VLForConditionalGeneration,
        "processor_class": AutoProcessor,
        "from_pretrained_kwargs": {"trust_remote_code": True},
    },
    "llava_onevision": {
        "model_class": LlavaOnevisionForConditionalGeneration,
        "processor_class": AutoProcessor,
        "from_pretrained_kwargs": {"trust_remote_code": True},
    },
    "llama": {
        "model_class": LlamaForCausalLM,
        "processor_class": LlamaTokenizer,
        "from_pretrained_kwargs": {"trust_remote_code": True},
    },
    "t5": {
        "model_class": T5ForConditionalGeneration,
        "processor_class": AutoTokenizer,
        "from_pretrained_kwargs": {},
    },
    "instructblip": {
        "model_class": InstructBlipForConditionalGeneration,
        "processor_class": AutoProcessor,
        "from_pretrained_kwargs": {"trust_remote_code": True},
    },
    "qwen": {
        "model_class": AutoModelForCausalLM,
        "processor_class": AutoTokenizer,
        "from_pretrained_kwargs": {},
    },
}


def get_tokenizer(
    tokenizer_or_processor: AutoProcessor | AutoTokenizer,
) -> AutoTokenizer:
    """从 Processor 或 Tokenizer 中获取底层的 Tokenizer 对象"""
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def load_model_for_inference(
    model_type: str,
    ckpt_path: str,
    use_lora: bool,
    model_path: str | None = None,
    device: torch.device | str | None = None,  # <-- 2. 添加 device 参数
) -> tuple[Any, AutoProcessor | AutoTokenizer]:
    """
    为推理（测试/生成）加载模型和分词器。

    该函数处理两种情况：
    1. 加载一个完整的、经过微调的模型。
    2. 加载一个基础模型，并应用其上的LoRA权重（权重会自动合并以提高推理速度）。

    它还会根据检查点中的元信息自动处理词汇表扩展。

    (DDP-safe): 接受一个 'device' 参数，以强制模型加载到特定GPU，
    覆盖 'device_map="auto"'，从而避免与DDP冲突。

    Args:
    ----
        model_type (str): 模型类型，例如 "qwen2_vl"。
        ckpt_path (str): 检查点路径（全量微调模型或LoRA适配器）。
        use_lora (bool): 是否加载和合并 LoRA 权重。
        model_path (str): 基础模型路径（仅在use_lora=True时需要）。
        device (torch.device | str | None): [DDP专用]
            如果提供，模型将被强制加载到此特定设备 (例如 "cuda:0")。

    Returns:
    -------
        tuple[Any, Union[AutoProcessor, AutoTokenizer]]: (加载和配置好的模型, 对应的处理器或分词器).

    """
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"不支持的模型类型: {model_type}")

    print(f"为推理加载 {model_type.upper()} 模型...")

    config = MODEL_CONFIG[model_type]
    model_class = config["model_class"]
    processor_class = config["processor_class"]
    from_pretrained_kwargs = config.get("from_pretrained_kwargs", {})

    # 1. 加载 Processor / Tokenizer
    print(f"从 '{ckpt_path}' 加载处理器/分词器...")
    processor = processor_class.from_pretrained(
        ckpt_path,
        padding_side="left",
        use_fast=True,
        **from_pretrained_kwargs,
    )
    tokenizer = get_tokenizer(processor)
    print(f"词汇表大小: {len(tokenizer)}")

    # ======================================================
    #  ⬇️  修改点 3: DDP 设备映射逻辑
    # ======================================================
    if device:
        # DDP 模式: 强制模型加载到 'device' (例如 'cuda:0' 或 'cuda:1')
        # {"": device} 告诉 accelerate 将所有模块加载到这个特定设备
        device_map = {"": device}
        print(f"[DDP] 强制模型加载到设备: {device}")
    else:
        # 原始行为: 自动拆分模型到所有可用 GPU (适用于单进程)
        device_map = "auto"
        print("[Single-Process] 使用 device_map='auto'")
    # ======================================================

    if use_lora:
        if not model_path:
            error_string = "使用LoRA时必须提供 'model_path'（基础模型路径）。"
            raise ValueError(error_string)

        # 2. 加载基础模型
        print(f"从 '{model_path}' 加载基础模型...")
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            **from_pretrained_kwargs,
        )
        print("基础模型加载完成。")

        # 3. 调整词汇表大小以匹配训练时的扩展
        # LoRA训练时会扩展词汇表，我们需要恢复这个扩展
        # 检查adapter_config.json中的信息
        adapter_config_path = os.path.join(ckpt_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            # PEFT会在modules_to_save中保存修改过的embedding层信息
            if adapter_config.get("modules_to_save"):
                print(f"检测到保存的模块: {adapter_config['modules_to_save']}")
                # 词汇表大小应该已经在tokenizer中正确设置了
                new_vocab_size = len(tokenizer)
                print(f"调整模型词汇表大小为: {new_vocab_size}")
                model.resize_token_embeddings(new_vocab_size)
                model.config.vocab_size = new_vocab_size

        # 4. 加载并合并LoRA权重
        print(f"加载LoRA权重从: {ckpt_path}")
        model = PeftModel.from_pretrained(
            model,
            ckpt_path,
            is_trainable=False,  # 推理模式
        )
        print("LoRA权重加载完成。")

        # 5. 合并权重以提高推理速度（可选）
        print("合并LoRA权重到基础模型...")
        model = model.merge_and_unload()
        print("LoRA权重合并完成。")
    else:
        # 2. 直接加载完整模型
        print(f"从 '{ckpt_path}' 加载完整模型...")
        model = model_class.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            **from_pretrained_kwargs,
        )
        print("完整模型加载完成。")

    # 验证词汇表大小匹配
    final_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != final_vocab_size:
        print(
            f"警告: Tokenizer大小 {len(tokenizer)} != 模型词汇表大小 {final_vocab_size}"
        )

    model.eval()
    print("模型加载完成并已设置为评估模式。")

    return model, processor


def _load_processor_and_tokenizer(args, config, base_model_path, local_rank, log_func):
    """加载处理器和分词器"""
    processor_class = config["processor_class"]
    from_pretrained_kwargs = config.get("from_pretrained_kwargs", {})

    if local_rank == 0:
        log_func(f"从 '{base_model_path}' 加载处理器/分词器...")

    # 特殊处理文本模型
    if args.model_type in ["qwen2", "qwen2_5", "llama", "qwen"]:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        processor = tokenizer
    else:
        processor = processor_class.from_pretrained(
            base_model_path,
            use_fast=True,
            **from_pretrained_kwargs,
        )
        tokenizer = processor.tokenizer

    return processor, tokenizer


def _load_base_model(args, config, base_model_path, config_obj, local_rank, log_func):
    """加载基础模型"""
    model_class = config["model_class"]
    from_pretrained_kwargs = config.get("from_pretrained_kwargs", {})

    if local_rank == 0:
        log_func(f"从 '{base_model_path}' 加载基础模型...")

    # 构建模型加载参数
    model_kwargs = {
        "config": config_obj,
        "trust_remote_code": True,
        **from_pretrained_kwargs,
    }

    # 根据训练参数设置数据类型
    if hasattr(args, "bf16") and args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif hasattr(args, "fp16") and args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    # 对于多模态模型，添加attention实现
    if args.model_type in ["qwen2_vl", "qwen2_5_vl", "llava_onevision"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = model_class.from_pretrained(base_model_path, **model_kwargs)
    return model


def _extend_vocabulary(
    args, model, tokenizer, new_tokens, local_rank, log_func, logger=None
):
    """扩展词汇表"""
    # 获取新tokens
    if new_tokens is None:
        train_data, _ = load_datasets(args, logger, local_rank)
        new_tokens = train_data.datasets[0].get_new_tokens()

    if local_rank == 0:
        log_func(f"从数据集中获取到 {len(new_tokens)} 个新 token 用于扩展词汇表。")

    original_vocab_size = len(tokenizer)
    tokenizer.add_tokens(new_tokens, special_tokens=False)
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)

    # 更新配置中的词汇表大小
    if args.model_type != "llava_onevision":
        model.config.vocab_size = new_vocab_size

    if local_rank == 0:
        log_func(f"原始词汇表大小: {original_vocab_size}")
        log_func(f"扩展后新词汇表大小: {new_vocab_size}")

    return original_vocab_size, new_vocab_size, new_tokens


def _save_token_metadata(
    args, original_vocab_size, new_vocab_size, new_tokens, local_rank
):
    """保存词汇表元数据"""
    ensure_dir(args.output_dir)
    if local_rank == 0:
        with open(os.path.join(args.output_dir, "token_meta.json"), "w") as f:
            json.dump(
                {
                    "original_vocab_size": original_vocab_size,
                    "new_vocab_size": new_vocab_size,
                    "added_tokens": len(new_tokens),
                },
                f,
                indent=2,
            )


def _setup_lora(args, model, local_rank, log_func):
    """配置LoRA（如果启用）"""
    if not (hasattr(args, "use_lora") and args.use_lora):
        return model

    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
    )

    if local_rank == 0:
        log_func("启用LoRA训练...")

    # 解析target_modules和modules_to_save
    target_modules = args.lora_target_modules.split(",")
    modules_to_save = (
        args.lora_modules_to_save.split(",")
        if hasattr(args, "lora_modules_to_save") and args.lora_modules_to_save
        else ["embed_tokens", "lm_head"]
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # 加载checkpoint权重（如果存在）
    _load_lora_checkpoint(args, model, local_rank, log_func)

    if local_rank == 0:
        model.print_trainable_parameters()

    return model


def _load_lora_checkpoint(args, model, local_rank, log_func):
    """加载LoRA检查点权重"""
    if not (hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint):
        return

    from peft import set_peft_model_state_dict

    checkpoint_name = os.path.join(
        args.resume_from_checkpoint, "adapter_model.safetensors"
    )
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")

    if os.path.exists(checkpoint_name):
        if local_rank == 0:
            log_func(f"从检查点加载LoRA权重: {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name, map_location="cpu")
        model = set_peft_model_state_dict(model, adapters_weights)
    elif local_rank == 0:
        log_func(f"未找到检查点: {checkpoint_name}")


def _freeze_only_embeddings(model, local_rank, log_func):
    """只训练embedding，冻结其他所有参数"""
    # 先冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 然后只解冻embedding相关参数
    embedding_unfrozen = False

    # 尝试不同的embedding参数路径
    embedding_paths = [
        (
            "language_model.embed_tokens",
            lambda m: getattr(m.language_model, "embed_tokens", None)
            if hasattr(m, "language_model")
            else None,
        ),
        ("embed_tokens", lambda m: getattr(m, "embed_tokens", None)),
        (
            "model.embed_tokens",
            lambda m: getattr(m.model, "embed_tokens", None)
            if hasattr(m, "model")
            else None,
        ),
    ]

    for path_name, path_getter in embedding_paths:
        embed_module = path_getter(model)
        if embed_module is not None:
            for name, param in embed_module.named_parameters():
                param.requires_grad = True
            if local_rank == 0:
                log_func(f"解冻 {path_name} 参数")
            embedding_unfrozen = True
            break

    # 尝试解冻lm_head参数（通常embedding和lm_head共享权重）
    lm_head_paths = [
        ("lm_head", lambda m: getattr(m, "lm_head", None)),
        (
            "language_model.lm_head",
            lambda m: getattr(m.language_model, "lm_head", None)
            if hasattr(m, "language_model")
            else None,
        ),
    ]

    for path_name, path_getter in lm_head_paths:
        lm_head_module = path_getter(model)
        if lm_head_module is not None:
            for name, param in lm_head_module.named_parameters():
                param.requires_grad = True
            if local_rank == 0:
                log_func(f"解冻 {path_name} 参数")
            break

    if not embedding_unfrozen and local_rank == 0:
        log_func("警告: 未找到embedding参数，请检查模型结构")

    if local_rank == 0:
        log_func("只训练embedding参数，冻结其他所有参数")


def _apply_freeze_strategy(
    args, model, original_vocab_size, local_rank, log_func, logger
):
    """应用参数冻结策略"""
    embedding_hooks = []

    if not hasattr(args, "freeze"):
        return embedding_hooks

    if args.freeze == "only_embeddings":
        _freeze_only_embeddings(model, local_rank, log_func)
    else:
        # 冻结视觉模型参数
        if args.freeze in ["visual", "all"]:
            if hasattr(model, "visual"):
                for name, param in model.visual.named_parameters():
                    param.requires_grad = False
                if local_rank == 0:
                    log_func("冻结视觉模型参数")
            if hasattr(model, "visual") and hasattr(model.visual, "merger"):
                for name, param in model.visual.merger.named_parameters():
                    param.requires_grad = False
                if local_rank == 0:
                    log_func("冻结视觉模型融合层参数")

        # 冻结原始embedding参数
        if args.freeze in ["embeddings", "all"]:
            embedding_hooks = freeze_original_embeddings_with_hook(
                model, original_vocab_size, logger
            )

    return embedding_hooks


def load_model_for_training(
    args: argparse.Namespace,
    new_tokens: list[str] | None = None,
    local_rank: int = 0,
    logger=None,
    nonewtokens: bool = False,
) -> tuple[Any, AutoProcessor | AutoTokenizer, int, int, list[str], list]:
    """
    为训练加载模型、分词器，并进行必要的设置。

    该函数负责：
    1. 根据模型类型加载基础模型和对应的分词器/处理器。
    2. 根据传入的 new_tokens 扩展词汇表。
    3. 调整模型嵌入层的大小以适应新词汇表。
    4. 根据freeze配置选择性冻结参数。
    5. 如果启用了LoRA，则应用PeftConfig来包装模型。
    6. 保存词汇表元数据，供后续推理使用。

    Args:
    ----
        args (argparse.Namespace): 包含所有配置的统一对象。
        new_tokens (list[str], optional): 需要添加到词汇表的新token列表。如果为None，从数据集获取。
        local_rank (int): 当前进程的rank，用于分布式训练。
        logger: 日志记录器，如果为None则使用print。

    Returns:
    -------
        tuple: (model, processor, original_vocab_size, new_vocab_size, new_tokens, embedding_hooks)

    """
    model_type = args.model_type
    base_model_path = args.base_model
    log_func = logger.info if logger else print

    if model_type not in MODEL_CONFIG:
        raise ValueError(f"不支持的模型类型: {model_type}")

    if local_rank == 0:
        log_func(f"为训练加载 {model_type.upper()} 模型...")

    # 1. 获取配置
    config = MODEL_CONFIG[model_type]
    from transformers import AutoConfig

    config_obj = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    # 2. 加载处理器和分词器
    processor, tokenizer = _load_processor_and_tokenizer(
        args, config, base_model_path, local_rank, log_func
    )

    # 3. 加载基础模型
    model = _load_base_model(
        args, config, base_model_path, config_obj, local_rank, log_func
    )

    if nonewtokens:
        original_vocab_size = len(tokenizer)
        new_vocab_size = len(tokenizer)
    else:
        # 4. 扩展词汇表
        original_vocab_size, new_vocab_size, new_tokens = _extend_vocabulary(
            args, model, tokenizer, new_tokens, local_rank, log_func, logger
        )

        # 5. 保存词汇表元数据
        _save_token_metadata(
            args, original_vocab_size, new_vocab_size, new_tokens, local_rank
        )

    # 6. 配置LoRA（如果启用）
    model = _setup_lora(args, model, local_rank, log_func)

    # 7. 应用参数冻结策略
    embedding_hooks = _apply_freeze_strategy(
        args, model, original_vocab_size, local_rank, log_func, logger
    )

    # 8. 设置训练模式
    model.train()
    if local_rank == 0:
        log_func("模型加载完成并已设置为训练模式。")

    return (
        model,
        processor,
        original_vocab_size,
        new_vocab_size,
        new_tokens,
        embedding_hooks,
    )


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args: argparse.Namespace, logger=None, local_rank=0):
    """根据配置加载训练和验证数据集"""
    log_func = logger.info if logger else print

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), (
        "prompt sample number does not match task number"
    )
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), (
        "data sample number does not match task number"
    )

    if local_rank == 0:
        log_func(f"tasks: {tasks}")
        log_func(f"train prompt sample num: {train_prompt_sample_num}")
        log_func(f"train data sample num: {train_data_sample_num}")

    train_datasets = []
    valid_datasets = []
    for task, prompt_sample_num, data_sample_num in zip(
        tasks, train_prompt_sample_num, train_data_sample_num, strict=False
    ):
        dataset_list = args.dataset.split(",")
        for dataset in dataset_list:
            train_dataset, valid_dataset = None, None
            # 统一将配置对象传递给各个数据集
            if task.lower() == "seqrec":
                train_dataset = SeqRecDataset(
                    args,
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            elif task.lower() == "mmseqrec":
                train_dataset = MultimodalSeqRecDataset(
                    args,
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            elif task.lower() == "seqrec_without_id":
                train_dataset = SeqRectWithoutItemIDDataset_1(
                    args,
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            elif task.lower() == "seqrec_with_title":
                train_dataset = SeqRecWithTitleDataset(
                    args,
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            elif task.lower() in ["item2index", "index2item"]:
                train_dataset = ItemFeatDataset(
                    args,
                    task=task.lower(),
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )

            elif task.lower() == "fusionseqrec":
                train_dataset = FusionSeqRecDataset(
                    args,
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )

            elif task.lower() in ["mmitem2index", "mmindex2item"]:
                train_dataset = MultimodalDataset(
                    args,
                    task=task.lower(),
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
                # valid_dataset = MultimodalDataset(
                #     args,
                #     mode="valid",
                #     task=task.lower(),
                #     dataset=dataset,
                #     prompt_sample_num=prompt_sample_num,
                #     sample_num=data_sample_num,
                # )
                # print(
                #     f"Prepare MultimodalDataset for {task} - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}"
                # )

            elif task.lower() == "mmitemenrich":
                train_dataset = TextEnrichDataset(
                    args,
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
                # valid_dataset = TextEnrichDataset(
                #     args,
                #     mode="valid",
                #     dataset=dataset,
                #     prompt_sample_num=prompt_sample_num,
                #     sample_num=data_sample_num,
                # )
                # print(
                #     f"Prepare TextEnrichDataset for {task} - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}"
                # )

            elif task.lower() == "mmitemenrichwithoutid":
                train_dataset = TextEnrichWihtoutItemIDDataset(
                    args,
                    mode="train",
                    dataset=dataset,
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )

            if task.lower() == "seqrec":
                valid_dataset = SeqRecDataset(
                    args,
                    mode="valid",
                    dataset=dataset,
                    prompt_sample_num=args.valid_prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            elif task.lower() == "mmseqrec":
                valid_dataset = MultimodalSeqRecDataset(
                    args,
                    mode="valid",
                    dataset=dataset,
                    prompt_sample_num=args.valid_prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            elif task.lower() == "seqrec_without_id":
                valid_dataset = SeqRectWithoutItemIDDataset_1(
                    args,
                    mode="valid",
                    dataset=dataset,
                    prompt_sample_num=args.valid_prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            elif task.lower() in ["item2index", "index2item"]:
                valid_dataset = ItemFeatDataset(
                    args,
                    task=task.lower(),
                    mode="valid",
                    dataset=dataset,
                    prompt_sample_num=args.valid_prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            elif task.lower() in ["mmitem2index", "mmindex2item"]:
                valid_dataset = MultimodalDataset(
                    args,
                    task=task.lower(),
                    mode="valid",
                    dataset=dataset,
                    prompt_sample_num=args.valid_prompt_sample_num,
                    sample_num=data_sample_num,
                    logger=logger,
                    local_rank=local_rank,
                )
            if train_dataset:
                train_datasets.append(train_dataset)
                if local_rank == 0:
                    log_func(f"Task: {task} - train sample nums: {len(train_dataset)}")
            if valid_dataset:
                valid_datasets.append(valid_dataset)
                if local_rank == 0:
                    log_func(f"Task: {task} - valid sample nums: {len(valid_dataset)}")
        train_data = ConcatDataset(train_datasets)
        valid_data = ConcatDataset(valid_datasets)

    if local_rank == 0:
        log_func(f"Train sample nums: {len(train_data)}")
        log_func(f"Valid sample nums: {len(valid_data)}")
    return train_data, valid_data


def load_test_dataset(args: argparse.Namespace, logger=None, local_rank=0):
    """根据配置加载测试数据集"""
    dataset_list = args.dataset.split(",")
    for dataset in dataset_list:
        if args.test_task.lower() == "seqrec":
            test_data = SeqRecDataset(
                args,
                mode="test",
                dataset=dataset,
                sample_num=args.sample_num,
                logger=logger,
                local_rank=local_rank,
            )
        elif args.test_task.lower() == "mmseqrec":
            test_data = MultimodalSeqRecDataset(
                args,
                mode="test",
                dataset=dataset,
                sample_num=args.sample_num,
                logger=logger,
                local_rank=local_rank,
            )
        elif args.test_task.lower() == "fusionseqrec":
            test_data = FusionSeqRecDataset(
                args,
                mode="test",
                dataset=dataset,
                sample_num=args.sample_num,
                logger=logger,
                local_rank=local_rank,
            )
        elif args.test_task.lower() in ["item2index", "index2item"]:
            test_data = ItemFeatDataset(
                args,
                task=args.test_task.lower(),
                mode="test",
                dataset=dataset,
                sample_num=args.sample_num,
                logger=logger,
                local_rank=local_rank,
            )
        elif args.test_task.lower() in ["mmitem2index", "mmindex2item"]:
            test_data = MultimodalDataset(
                args,
                mode="test",
                dataset=dataset,
                task=args.test_task.lower(),
                sample_num=args.sample_num,
                logger=logger,
                local_rank=local_rank,
            )
        else:
            raise NotImplementedError

    return test_data


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def make_run_name(args: argparse.Namespace) -> str:
    """
    run_name='none' 时自动生成；格式：
    {base_model_last}__{dataset}__b{bs}__gc{0|1}__{tasks}__p{prompt_num}__idx{index_file_key}__{timestamp}
    index_file_key 为 index_file 去掉前缀 '.' 和后缀 '.json'（若存在）。
    """
    # if hasattr(args, "run_name") and not(args.run_name in ["None","none"]):
    #     return args.run_name

    base_name = os.path.basename(os.path.normpath(args.base_model))
    gc_flag = (
        "1"
        if hasattr(args, "use_gradient_checkpointing")
        and args.use_gradient_checkpointing
        else "0"
    )

    # 处理 index_file
    idx_file = (
        os.path.basename(args.index_file) if hasattr(args, "index_file") else "none"
    )
    if idx_file != "none":
        idx_file = idx_file.removeprefix(".index_")
        idx_file = idx_file.removesuffix(".json")
    idx_key = idx_file or "none"

    # 添加时间戳以确保唯一性
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    # 处理tasks和prompt_sample_num
    tasks = args.tasks if hasattr(args, "tasks") else "unknown"
    prompt_num = (
        args.train_prompt_sample_num
        if hasattr(args, "train_prompt_sample_num")
        else "0"
    )
    dataset = args.dataset if hasattr(args, "dataset") else "unknown"
    batch_size = (
        args.per_device_batch_size if hasattr(args, "per_device_batch_size") else "1"
    )

    method = "Lora" if args.use_lora else "Finetune"

    lr = args.learning_rate

    return f"{base_name}__{dataset}__{method}__lr{lr}__b{batch_size}__gc{gc_flag}__{tasks}__p{prompt_num}__idx{idx_key}__{timestamp}"


def verify_token_ordering(tokenizer_or_processor, original_vocab_size, new_tokens):
    """验证新添加的token是否真的在词汇表末尾"""
    from transformers import AutoProcessor

    if isinstance(tokenizer_or_processor, AutoProcessor):
        tokenizer = tokenizer_or_processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor

    print("=== 验证Token排序 ===")

    # 检查原始token的一些示例
    original_samples = [0, 1, 100, 1000, original_vocab_size - 1]
    print("原始token示例:")
    for token_id in original_samples:
        if token_id < original_vocab_size:
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            print(f"  ID {token_id}: '{token}'")

    # 检查新添加的token
    print(f"\n新添加的token (总共{len(new_tokens)}个):")
    current_vocab_size = len(tokenizer)
    new_token_start = original_vocab_size

    # 验证前几个新token
    for i, expected_token in enumerate(new_tokens[:5]):  # 只显示前5个
        token_id = new_token_start + i
        if token_id < current_vocab_size:
            actual_token = tokenizer.convert_ids_to_tokens([token_id])[0]
            print(
                f"  ID {token_id}: 期望 '{expected_token}' -> 实际 '{actual_token}' {'✓' if expected_token == actual_token else '✗'}"
            )

    # 验证最后几个新token
    if len(new_tokens) > 5:
        print("  ...")
        for i in range(max(0, len(new_tokens) - 3), len(new_tokens)):
            expected_token = new_tokens[i]
            token_id = new_token_start + i
            if token_id < current_vocab_size:
                actual_token = tokenizer.convert_ids_to_tokens([token_id])[0]
                print(
                    f"  ID {token_id}: 期望 '{expected_token}' -> 实际 '{actual_token}' {'✓' if expected_token == actual_token else '✗'}"
                )

    # 最终验证
    print("\n验证结果:")
    print(f"  原始词汇表大小: {original_vocab_size}")
    print(f"  当前词汇表大小: {current_vocab_size}")
    print(f"  新增token数量: {len(new_tokens)}")
    print(f"  预期新token ID范围: {original_vocab_size} ~ {current_vocab_size - 1}")

    return (
        new_token_start == original_vocab_size
        and current_vocab_size == original_vocab_size + len(new_tokens)
    )


def freeze_original_embeddings_with_hook(
    model: torch.nn.Module, original_vocab_size: int, logger=None
) -> list:
    """
    使用梯度hook冻结原始embedding参数，只训练新添加的token embedding

    Args:
        model: PyTorch模型
        original_vocab_size: 原始词汇表大小
        logger: 可选的日志记录器

    Returns:
        list: 注册的hook句柄列表，用于后续清理

    """
    hooks = []

    # 使用logger或print
    log_func = logger.info if logger else print

    def set_grads_to_zero_hook(grad: torch.Tensor) -> torch.Tensor:
        """梯度hook函数，将原始token的梯度置零"""
        if grad is not None:
            new_grad = grad.clone()
            new_grad[:original_vocab_size] = 0.0
            return new_grad
        return grad

    if hasattr(model, "language_model") and hasattr(
        model.language_model, "embed_tokens"
    ):
        embed_module = model.language_model.embed_tokens
        if hasattr(embed_module, "weight") and embed_module.weight.requires_grad:
            handle = embed_module.weight.register_hook(set_grads_to_zero_hook)
            hooks.append(handle)
            log_func(
                f"为 language_model.embed_tokens 注册hook, shape: {embed_module.weight.shape}"
            )
            log_func(f"冻结前 {original_vocab_size} 个token的梯度")
    # 3. 冻结视觉模型的rotary_emb
    # if hasattr(model, "language_model") and hasattr(
    #     model.language_model, "rotary_emb"
    # ):
    #     visual_rotary = model.language_model.rotary_emb
    #     if (
    #         hasattr(visual_rotary, "weight")
    #         and visual_rotary.weight.requires_grad
    #     ):
    #         visual_rotary.weight.requires_grad_(False)
    #         print("冻结 visual.rotary_pos_emb 参数")

    return hooks


# 1. 仅剔除“框架”特殊 token，保留 item token
FRAMEWORK_SPECIAL = {
    "<s>",
    "</s>",
    "<unk>",
    "<pad>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
}
# 2. 预编译空白符正则
WHITE_PAT = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    只去掉框架特殊 token 和空白，完全保留 <a_*> <b_*> <c_*> <d_*> 等用户 token
    """
    # 去掉框架特殊 token
    for tok in FRAMEWORK_SPECIAL:
        text = text.replace(tok, "")
    # 去掉所有空白（空格、换行、制表）
    text = WHITE_PAT.sub("", text)
    return text.strip()
