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
    Qwen2_5_VLForConditionalGeneration,
    T5ForConditionalGeneration,
)

from .data import (
    FusionSeqRecDataset,
    ItemFeatDataset,
    ItemSearchDataset,
    MultimodalDataset,
    PreferenceObtainDataset,
    SeqRecDataset,
    SeqRectWithoutItemIDDataset_1,
    SeqRecWithTitleDataset,
    TextEnrichDataset,
    TextEnrichWihtoutItemIDDataset,
)
from .type import Args

# ----------------- 模型和分词器加载工具 -----------------

MODEL_CONFIG = {
    "qwen_vl": {
        "model_class": Qwen2_5_VLForConditionalGeneration,
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
    model_type: str, model_path: str, ckpt_path: str, use_lora: bool
) -> tuple[Any, AutoProcessor | AutoTokenizer]:
    """
    为推理（测试/生成）加载模型和分词器。

    该函数处理两种情况：
    1. 加载一个完整的、经过微调的模型。
    2. 加载一个基础模型，并应用其上的LoRA权重（权重会自动合并以提高推理速度）。

    它还会根据检查点中的元信息自动处理词汇表扩展。

    Args:
    ----
        model_type (str): 模型类型，例如 "qwen_vl"。
        model_path (str): 基础或完整模型的路径。
        ckpt_path (str): LoRA 检查点的路径 (如果 use_lora 为 True)。
        use_lora (bool): 是否加载和合并 LoRA 权重。

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
    # 分词器/处理器应该始终从主模型路径加载。
    if ckpt_path == "":
        ckpt_path = model_path
    print(f"从 '{ckpt_path}' 加载分词器...")
    processor = processor_class.from_pretrained(
        ckpt_path, padding_side="left", **from_pretrained_kwargs
    )
    tokenizer = get_tokenizer(processor)
    print(f"词汇表大小: {len(tokenizer)}")
    if use_lora:
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
        # 3. 处理词汇表扩展
        # 对于推理，我们从保存的元数据中恢复词汇表大小
        # 如果是LoRA模型，则元数据在ckpt_path中；如果不是，则元数据应在模型路径本身
        token_meta_path_dir = ckpt_path
        token_meta_path = os.path.join(token_meta_path_dir, "token_meta.json")
        if os.path.exists(token_meta_path):
            with open(token_meta_path) as f:
                token_meta = json.load(f)
            new_vocab_size = token_meta["new_vocab_size"]
            old_vocab_size = token_meta["original_vocab_size"]
            print(
                f"从元数据中发现新词汇表大小: {new_vocab_size}, 旧词汇表大小: {old_vocab_size}"
            )
            model.resize_token_embeddings(new_vocab_size)
            model.config.vocab_size = new_vocab_size
        else:
            print("警告: 未找到 token_meta.json, 词汇表可能不匹配。")
        if not ckpt_path:
            raise ValueError("使用LoRA时必须提供 'ckpt_path'。")
        print(f"加载并合并LoRA权重从: {ckpt_path}")
        model = PeftModel.from_pretrained(model, ckpt_path)
        model = model.merge_and_unload()
        print("LoRA 权重合并完成。")
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

    final_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != final_vocab_size:
        print(
            f"Tokenizer size {len(tokenizer)} != model vocab size {final_vocab_size}"
        )

    model.eval()
    print("模型加载完成并已设置为评估模式。")

    return model, processor


def load_model_for_training(
    args: Args,
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
    model_args = args.global_args
    train_args = args.train_args

    model_type = model_args.model_type
    base_model_path = model_args.base_model

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
    ensure_dir(model_args.output_dir)
    with open(os.path.join(model_args.output_dir, "token_meta.json"), "w") as f:
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
    if train_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        print("启用LoRA训练...")
        # 冻结原始embedding
        # freeze_original_embeddings_simple(model, original_vocab_size)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_args.lora_r,
            lora_alpha=train_args.lora_alpha,
            lora_dropout=train_args.lora_dropout,
            target_modules=train_args.lora_target_modules.split(","),
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 将模型移动到设备
    model.to(train_args.device)
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


def load_datasets(args: Args):
    """根据配置加载训练和验证数据集"""
    dataset_args = args.dataset_args
    tasks = dataset_args.tasks.split(",")

    train_prompt_sample_num = [
        int(_) for _ in dataset_args.train_prompt_sample_num.split(",")
    ]
    assert len(tasks) == len(train_prompt_sample_num), (
        "prompt sample number does not match task number"
    )
    train_data_sample_num = [
        int(_) for _ in dataset_args.train_data_sample_num.split(",")
    ]
    assert len(tasks) == len(train_data_sample_num), (
        "data sample number does not match task number"
    )

    print("train prompt sample num:", train_prompt_sample_num)
    print("train data sample num:", train_data_sample_num)

    train_datasets = []
    for task, prompt_sample_num, data_sample_num in zip(
        tasks, train_prompt_sample_num, train_data_sample_num, strict=False
    ):
        # 统一将配置对象传递给各个数据集
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(
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
        elif task.lower() == "item2index" or task.lower() == "index2item":
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

        elif task.lower() == "itemsearch":
            dataset = ItemSearchDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "preferenceobtain":
            dataset = PreferenceObtainDataset(
                args,
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "mmitem2index" or task.lower() == "mmindex2item":
            dataset = MultimodalDataset(
                args,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

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
        else:
            raise NotImplementedError
        train_datasets.append(dataset)
    train_data = ConcatDataset(train_datasets)
    print("train sample nums:", len(train_data))

    # if task.lower() == "mmitemenrichwithoutid":
    # valid_data = TextEnrichWihtoutItemIDDataset(args, mode="valid", prompt_sample_num=dataset_args.valid_prompt_sample_num, sample_num=data_sample_num)
    # else:
    if task.lower() == "seqrec_without_id":
        valid_data = SeqRectWithoutItemIDDataset_1(
            args,
            mode="valid",
            prompt_sample_num=dataset_args.valid_prompt_sample_num,
            sample_num=data_sample_num,
        )
    else:
        valid_data = SeqRecDataset(
            args,
            mode="valid",
            prompt_sample_num=dataset_args.valid_prompt_sample_num,
            sample_num=data_sample_num,
        )
    print("valid sample nums:", len(valid_data))
    return train_data, valid_data


def load_test_dataset(args: Args):
    """根据配置加载测试数据集"""
    test_args = args.test_args
    if test_args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(
            args,
            mode="test",
            sample_num=test_args.sample_num,
        )
    elif test_args.test_task.lower() == "itemsearch":
        test_data = ItemSearchDataset(
            args, mode="test", sample_num=test_args.sample_num
        )
    elif test_args.test_task.lower() == "fusionseqrec":
        test_data = FusionSeqRecDataset(
            args, mode="test", sample_num=test_args.sample_num
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


def freeze_original_embeddings_simple(model, original_vocab_size):
    """简单版本：直接冻结原有embedding参数"""
    input_embeddings = model.get_input_embeddings()

    # 冻结原有token的embedding
    with torch.no_grad():
        # 将原有embedding参数的requires_grad设为False
        original_embeddings = input_embeddings.weight[:original_vocab_size]
        original_embeddings.requires_grad_(False)

        # 确保新增embedding参数可以训练
        new_embeddings = input_embeddings.weight[original_vocab_size:]
        new_embeddings.requires_grad_(True)

    print(f"冻结了前 {original_vocab_size} 个token的embedding参数")
    print(
        f"保持后 {len(input_embeddings.weight) - original_vocab_size} 个新token的embedding可训练"
    )


def freeze_original_embeddings_for_lora(model, original_vocab_size):
    """适用于LoRA+modules_to_save的选择性冻结"""
    # 找到modules_to_save中的embed_tokens
    for name, module in model.named_modules():
        if "embed_tokens" in name and hasattr(module, "weight"):
            print(f"找到embedding层: {name}, shape: {module.weight.shape}")

            # 冻结原始token (0 到 original_vocab_size-1)
            module.weight[:original_vocab_size].requires_grad_(False)

            # 保持新token可训练 (original_vocab_size 到 end)
            module.weight[original_vocab_size:].requires_grad_(True)

            print(f"冻结了前 {original_vocab_size} 个token")
            print(
                f"保持后 {module.weight.shape[0] - original_vocab_size} 个新token可训练"
            )
            break


def freeze_original_embeddings_with_hook(model, original_vocab_size):
    """
    使用梯度hook冻结原始embedding参数
    """
    hooks = []

    def create_embedding_hook(vocab_size):
        def hook_fn(grad):
            if grad is not None:
                # 创建新的梯度，原始token位置置零
                new_grad = grad.clone()
                new_grad[:vocab_size] = 0.0
                return new_grad
            return grad

        return hook_fn

    # 为embedding注册hook
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "embed_tokens" in name:
                handle = param.register_hook(
                    create_embedding_hook(original_vocab_size)
                )
                hooks.append(handle)
                print(f"为 {name} 注册embedding hook, shape: {param.shape}")

    print(
        f"注册了 {len(hooks)} 个梯度hook来冻结前 {original_vocab_size} 个token"
    )
    return hooks
