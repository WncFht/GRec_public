import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..collator import MultiModalCollator
from ..config import parse_args
from ..data import SeqRectWithoutItemIDDataset_1


def main():
    # 配置
    ckpt_dir = os.environ.get("CKPT_PATH")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载参数
    args = parse_args()
    args.dataset_args.dataset = "Instruments"

    # 加载tokenizer和模型
    processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_dir, trust_remote_code=True
    ).to(device)
    model.eval()

    # 加载数据集（只取前2个样本）
    dataset = SeqRectWithoutItemIDDataset_1(args, mode="test", sample_num=2)
    # 使用 UnifiedTestCollator 适配 Qwen2.5-VL
    collator = MultiModalCollator(args, processor)
    tokenizer = processor.tokenizer
    for i in range(len(dataset)):
        batch = [dataset[i]]
        inputs = collator(batch)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"\n==== Sample {i} ====")
        print("Input:")
        print(batch[0].input_text)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Decoded Output:")
        print(decoded)


if __name__ == "__main__":
    main()
