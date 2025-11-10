from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    InstructBlipForConditionalGeneration,
    LlamaForCausalLM,
    LlavaOnevisionForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    T5ForConditionalGeneration,
)

MODEL_FACTORY = {
    # "qwen2_5_vl":     (Qwen2_5_VLForConditionalGeneration, AutoProcessor),
    "qwen2_vl": (Qwen2VLForConditionalGeneration, AutoProcessor),
    "llava_onevision": (LlavaOnevisionForConditionalGeneration, AutoProcessor),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "instructblip": (InstructBlipForConditionalGeneration, AutoProcessor),
    "qwen": (AutoModelForCausalLM, AutoTokenizer),
}


def get_model_class_and_processor(
    model_type: str,
) -> tuple[Any, Any]:
    """
    返回 (model_class, processor_or_tokenizer_class)
    统一接口：processor_class 对于 LLM 其实就是 tokenizer
    """
    if model_type not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return MODEL_FACTORY[model_type]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="基础模型目录")
    p.add_argument("--lora_path", required=True, help="LoRA 权重目录")
    p.add_argument("--save_path", required=True, help="合并后模型保存目录")
    p.add_argument("--model_type", default="qwen2_vl")
    return p.parse_args()


def merge_lora_and_save(
    base_model: str,
    lora_path: str,
    save_path: str,
    model_type: str = "qwen2_vl",
):
    """把 LoRA 权重合并进基础模型并保存"""
    # 1. 动态拿到 model_class / processor_class
    model_class, processor_class = get_model_class_and_processor(model_type)

    # 2. 加载 tokenizer / processor（词汇表扩展已随 LoRA 目录）
    processor = processor_class.from_pretrained(lora_path)
    tokenizer = (
        processor if hasattr(processor, "vocab_size") else processor.tokenizer
    )
    new_vocab_size = len(tokenizer)
    print(f"[INFO] 词汇表大小：{new_vocab_size}")
    # 3. 加载基础模型
    model = model_class.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. 先挂载 LoRA（此时模型权重仍是原词表）
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)

    # 5. 再扩充词表 → PEFT 会自动把新 embedding 的 LoRA 权重一起 resize
    if new_vocab_size != model.config.vocab_size:
        model.resize_token_embeddings(new_vocab_size)
        model.config.vocab_size = new_vocab_size
        print(f"[INFO] 已调整模型词汇表为 {new_vocab_size}")

    # 6. 合并
    model = model.merge_and_unload()
    # 6. 保存
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)
    processor.save_pretrained(save_path)
    print(f"[INFO] 合并后模型已保存到 {save_path}")


if __name__ == "__main__":
    args = parse_args()
    merge_lora_and_save(
        base_model=args.base_model,
        lora_path=args.lora_path,
        save_path=args.save_path,
        model_type=args.model_type,
    )
