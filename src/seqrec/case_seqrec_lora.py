import argparse

import torch

from src.collator import UnifiedTestCollator
from src.data import SeqRecDataset
from src.parser import parse_dataset_args, parse_global_args, parse_test_args
from src.utils import load_model_for_inference


def main(args: argparse.Namespace):
    """使用LoRA模型进行案例测试"""
    dataset = SeqRecDataset(args, mode="test")
    print(f"测试数据集大小: {len(dataset)}")
    print("样例数据:", dataset[0])

    # 使用load_model_for_inference加载LoRA模型
    model, processor = load_model_for_inference(
        model_type=args.model_type,
        ckpt_path=args.ckpt_path,
        use_lora=args.lora,
        model_path=args.base_model if args.lora else None,
    )
    
    collator = UnifiedTestCollator(args, processor_or_tokenizer=processor)
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(model, "device"):
        model.to(device)

    length = len(dataset)
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
                max_new_tokens=4,
                num_beams=10,
                num_return_sequences=10,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
        
        output_ids = output["sequences"]
        scores = output["sequences_scores"]

        # 解码所有结果
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        output_texts = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        print("生成结果:")
        # 提取"Response:"后面的输出
        for j, (text, score) in enumerate(zip(output_texts, scores)):
            response = text.split("Response:")[-1].strip()
            print(f"  {j+1}. {response} | 分数: {float(score):.4f}")
            
        # 检查是否命中
        responses = [text.split("Response:")[-1].strip() for text in output_texts]
        if target_text in responses:
            print(f"✓ 命中! 排名: {responses.index(target_text) + 1}")
        else:
            print("✗ 未命中")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA模型案例测试")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    
    args = parser.parse_args()
    main(args)