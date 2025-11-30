import argparse
import re

import torch

from src.collator import (
    ChatTemplateTestCollator,
    TestCollator,
    UnifiedTestCollator,
)
from src.parser import parse_dataset_args, parse_global_args, parse_test_args
from src.utils import load_model_for_inference, load_test_dataset


def main(args: argparse.Namespace):
    dataset = load_test_dataset(args)
    print(f"测试数据集大小: {len(dataset)}")
    print("样例数据:", dataset[0])

    # 使用load_model_for_inference加载LoRA模型
    model, processor = load_model_for_inference(
        model_type=args.model_type,
        ckpt_path=args.ckpt_path,
        use_lora=args.lora,
        model_path=args.base_model if args.lora else None,
    )

    if args.model_type == "llama":
        collator = TestCollator(args, tokenizer=processor)
        split_word = "Response: "
    elif args.model_type in ["qwen"]:
        collator = ChatTemplateTestCollator(args, tokenizer=processor)
        split_word = "assistant"
    else:
        collator = UnifiedTestCollator(args, processor_or_tokenizer=processor)
        split_word = "assistant"

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(model, "device"):
        model.to(device)

    length = len(dataset)

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

    # 测试最后5个样本
    for i in range(max(0, length - 5), length):
        batch = collator([dataset[i]])
        inputs = batch[0]
        target_text = batch[1]

        print("=" * 80)
        print(f"测试样本 {i}:")
        print(f"目标: {target_text}")
        print("-" * 40)

        # 将输入移到GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 使用beam search生成
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=1024,
                num_beams=2,
                # num_return_sequences=2,
                output_scores=True,
                return_dict_in_generate=True,
                # early_stopping=True,
            )

        output_ids = output["sequences"]
        scores = output["sequences_scores"]

        # 解码所有结果
        tokenizer = (
            processor.tokenizer
            if hasattr(processor, "tokenizer")
            else processor
        )
        output_texts = tokenizer.batch_decode(
            output_ids, skip_special_tokens=False
        )

        print("生成结果:")
        # 提取"Response:"后面的输出
        for j, (text, score) in enumerate(
            zip(output_texts, scores, strict=False)
        ):
            response = clean_text(text)
            # response = clean_text(text.split(split_word)[-1])
            print(
                f"  {j + 1}. {response} | ",
                target_text[0],
                f"分数: {float(score):.4f}",
            )

        # 检查是否命中
        responses = [
            clean_text(text.split(split_word)[-1]) for text in output_texts
        ]
        # remove the blank in responses
        if target_text[0] in responses:
            print(f"✓ 命中! 排名: {responses.index(target_text[0]) + 1}")
        else:
            print("✗ 未命中")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    main(args)
