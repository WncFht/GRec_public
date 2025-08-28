import argparse
import datetime
import json
import os
import random
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

from .data import (
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
) -> tuple[Any, AutoProcessor | AutoTokenizer]:
    """
    为推理（测试/生成）加载模型和分词器。

    该函数处理两种情况：
    1. 加载一个完整的、经过微调的模型。
    2. 加载一个基础模型，并应用其上的LoRA权重（权重会自动合并以提高推理速度）。

    它还会根据检查点中的元信息自动处理词汇表扩展。

    Args:
    ----
        model_type (str): 模型类型，例如 "qwen2_vl"。
        ckpt_path (str): 检查点路径（全量微调模型或LoRA适配器）。
        use_lora (bool): 是否加载和合并 LoRA 权重。
        model_path (str): 基础模型路径（仅在use_lora=True时需要）。

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
    # 对于LoRA模型，从ckpt_path加载（包含扩展的词表）
    # 对于全量微调，也从ckpt_path加载
    print(f"从 '{ckpt_path}' 加载处理器/分词器...")
    processor = processor_class.from_pretrained(
        ckpt_path,
        padding_side="left",
        use_fast=True,
        **from_pretrained_kwargs,
    )
    tokenizer = get_tokenizer(processor)
    print(f"词汇表大小: {len(tokenizer)}")
    
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
            device_map="auto",
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
            if "modules_to_save" in adapter_config and adapter_config["modules_to_save"]:
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
            is_trainable=False  # 推理模式
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
            device_map="auto",
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


def load_model_for_training(
    args: argparse.Namespace,
    new_tokens: list[str],
) -> tuple[Any, AutoProcessor | AutoTokenizer]:
    """
    为训练加载模型、分词器，并进行必要的设置。

    该函数负责：
    1. 根据模型类型加载基础模型和对应的分词器/处理器。
    2. 根据传入的 new_tokens 扩展词汇表。
    3. 调整模型嵌入层的大小以适应新词汇表。
    4. 冻结原始词汇表的嵌入，以便只训练新添加的token。
    5. 如果启用了LoRA，则应用PeftConfig来包装模型。
    6. 保存词汇表元数据，供后续推理使用。

    Attributes
    ----------
        args (Args): 包含所有配置的统一对象。
        new_tokens (list[str]): 从数据集中提取的、需要添加到词汇表的新token列表。

    Returns
    -------
        tuple[Any, Union[AutoProcessor, AutoTokenizer]]: (加载和配置好的模型, 对应的处理器或分词器).

    """
    model_type = args.model_type
    base_model_path = args.base_model

    if model_type not in MODEL_CONFIG:
        raise ValueError(f"不支持的模型类型: {model_type}")

    print(f"为训练加载 {model_type.upper()} 模型...")

    config = MODEL_CONFIG[model_type]
    model_class = config["model_class"]
    processor_class = config["processor_class"]
    from_pretrained_kwargs = config.get("from_pretrained_kwargs", {})

    # 1. 加载 Processor / Tokenizer
    print(f"从 '{base_model_path}' 加载分词器...")

    # 根据模型类型设置正确的 padding_side
    if model_type in ["qwen_vl", "llama", "qwen"]:
        # Decoder-only 模型需要 left padding 以保持因果性
        from_pretrained_kwargs["padding_side"] = "left"
    elif model_type in ["t5", "instructblip"]:
        # Encoder-decoder 模型可以使用 right padding
        from_pretrained_kwargs["padding_side"] = "right"

    processor = processor_class.from_pretrained(
        base_model_path, **from_pretrained_kwargs
    )
    tokenizer = get_tokenizer(processor)
    original_vocab_size = len(tokenizer)
    print(f"原始词汇表大小: {original_vocab_size}")

    # 2. 加载基础模型
    print(f"从 '{base_model_path}' 加载基础模型...")
    model = model_class.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **from_pretrained_kwargs,
    )

    # 3. 扩展词汇表
    print(f"从数据集中获取到 {len(new_tokens)} 个新 token 用于扩展词汇表。")
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    new_vocab_size = len(tokenizer)
    print(f"扩展后新词汇表大小: {new_vocab_size}")

    # 调整模型嵌入层大小
    model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size

    # 验证并保存词汇表元数据
    ensure_dir(args.output_dir)
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

    # 4. 根据LoRA配置包装模型
    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        print("启用LoRA训练...")
        # 冻结原始embedding
        # freeze_original_embeddings_simple(model, original_vocab_size)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 将模型移动到设备
    model.to(args.device)
    model.train()
    print("模型加载完成并已设置为训练模式。")

    return model, processor


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args: argparse.Namespace):
    """根据配置加载训练和验证数据集"""
    tasks = args.tasks.split(",")

    train_prompt_sample_num = [
        int(_) for _ in args.train_prompt_sample_num.split(",")
    ]
    assert len(tasks) == len(train_prompt_sample_num), (
        "prompt sample number does not match task number"
    )
    train_data_sample_num = [
        int(_) for _ in args.train_data_sample_num.split(",")
    ]
    assert len(tasks) == len(train_data_sample_num), (
        "data sample number does not match task number"
    )

    print("tasks:", tasks)
    print("train prompt sample num:", train_prompt_sample_num)
    print("train data sample num:", train_data_sample_num)

    train_datasets = []
    valid_datasets = []
    for task, prompt_sample_num, data_sample_num in zip(
        tasks, train_prompt_sample_num, train_data_sample_num, strict=False
    ):
        dataset, valid_dataset = None, None
        # 统一将配置对象传递给各个数据集
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )
        elif task.lower() == "mmseqrec":
            dataset = MultimodalSeqRecDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )
        elif task.lower() == "seqrec_without_id":
            dataset = SeqRectWithoutItemIDDataset_1(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )
        elif task.lower() == "seqrec_with_title":
            dataset = SeqRecWithTitleDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )
        elif task.lower() in ["item2index", "index2item"]:
            dataset = ItemFeatDataset(
                args,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "fusionseqrec":
            dataset = FusionSeqRecDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() in ["mmitem2index", "mmindex2item"]:
            dataset = MultimodalDataset(
                args,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )
            print("Prepare MultimodalDataset num:", len(dataset))

        elif task.lower() == "mmitemenrich":
            dataset = TextEnrichDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "mmitemenrichwithoutid":
            dataset = TextEnrichWihtoutItemIDDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        if task.lower() == "seqrec":
            valid_dataset = SeqRecDataset(
                args,
                mode="valid",
                prompt_sample_num=args.valid_prompt_sample_num,
                sample_num=data_sample_num,
            )
        elif task.lower() == "mmseqrec":
            valid_dataset = MultimodalSeqRecDataset(
                args,
                mode="valid",
                prompt_sample_num=args.valid_prompt_sample_num,
                sample_num=data_sample_num,
            )
        elif task.lower() == "seqrec_without_id":
            valid_dataset = SeqRectWithoutItemIDDataset_1(
                args,
                mode="valid",
                prompt_sample_num=args.valid_prompt_sample_num,
                sample_num=data_sample_num,
            )
        if dataset:
            train_datasets.append(dataset)
        if valid_dataset:
            valid_datasets.append(valid_dataset)
    train_data = ConcatDataset(train_datasets)
    valid_data = ConcatDataset(valid_datasets)

    print("Train sample nums:", len(train_data))
    print("Valid sample nums:", len(valid_data))
    return train_data, valid_data


def load_test_dataset(args: argparse.Namespace):
    """根据配置加载测试数据集"""
    if args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(
            args,
            mode="test",
            sample_num=args.sample_num,
        )
    elif args.test_task.lower() == "fusionseqrec":
        test_data = FusionSeqRecDataset(
            args, mode="test", sample_num=args.sample_num
        )
    else:
        raise NotImplementedError

    return test_data


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def verify_token_ordering(
    tokenizer_or_processor, original_vocab_size, new_tokens
):
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
    print(
        f"  预期新token ID范围: {original_vocab_size} ~ {current_vocab_size - 1}"
    )

    return (
        new_token_start == original_vocab_size
        and current_vocab_size == original_vocab_size + len(new_tokens)
    )


def freeze_original_embeddings_with_hook(
    model: torch.nn.Module, original_vocab_size: int
) -> list:
    """
    使用梯度hook冻结原始embedding参数，只训练新添加的token embedding

    Args:
        model: PyTorch模型
        original_vocab_size: 原始词汇表大小

    Returns:
        list: 注册的hook句柄列表，用于后续清理

    """
    hooks = []

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
        if (
            hasattr(embed_module, "weight")
            and embed_module.weight.requires_grad
        ):
            handle = embed_module.weight.register_hook(set_grads_to_zero_hook)
            hooks.append(handle)
            print(
                f"为 language_model.embed_tokens 注册hook, shape: {embed_module.weight.shape}"
            )
            print(f"冻结前 {original_vocab_size} 个token的梯度")
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
