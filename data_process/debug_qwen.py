import json
import os

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def debug_item_3186():
    """专门调试第3186个物品"""
    # 配置
    dataset = "Instruments"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.abspath(os.path.join("data", dataset))
    target_item = 575

    print(f"=== 调试物品 #{target_item} ===")
    print(f"Device: {device}")

    # 1. 加载数据集信息
    print("\n1. 加载数据集信息...")
    item2id_path = os.path.join(dataset_path, f"{dataset}.item2id")
    id2item = {}
    with open(item2id_path) as f:
        for line in f:
            item_id, num_id = line.strip().split("\t")
            id2item[int(num_id)] = item_id

    item_info_path = os.path.join(dataset_path, f"{dataset}.item.json")
    with open(item_info_path) as f:
        item_info = json.load(f)

    # 2. 获取目标物品信息
    print("\n2. 获取目标物品信息...")
    if str(target_item) not in item_info:
        print(f"错误：物品 {target_item} 不存在！")
        return

    item_data = item_info[str(target_item)]
    item_id = id2item[target_item]

    print(f"物品ID: {item_id}")
    print(f"物品数据字段: {list(item_data.keys())}")

    # 打印物品详细信息
    for key, value in item_data.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")

    # 3. 构造文本
    print("\n3. 构造文本...")
    text_parts = []
    if item_data.get("title"):
        text_parts.append(f"Title: {item_data['title']}")
    if item_data.get("brand"):
        text_parts.append(f"Brand: {item_data['brand']}")
    if item_data.get("categories"):
        text_parts.append(f"Categories: {item_data['categories']}")
    if item_data.get("description") and item_data["description"].strip():
        text_parts.append(f"Description: {item_data['description']}")

    text = " | ".join(text_parts)
    print(f"文本长度: {len(text)}")
    print(f"文本内容: {text[:200]}...")

    # 检查文本异常字符
    special_chars = [char for char in text if ord(char) > 65535]
    if special_chars:
        print(f"警告：发现特殊字符: {special_chars[:10]}")

    # 4. 加载图片
    print("\n4. 加载图片...")
    image_path = os.path.join(dataset_path, "images", f"{item_id}.jpg")
    print(f"图片路径: {image_path}")
    print(f"图片存在: {os.path.exists(image_path)}")

    image = None
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"图片尺寸: {image.size}")
        except Exception as e:
            print(f"图片加载失败: {e}")

    # 5. 加载模型
    print("\n5. 加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    # 6. 处理输入
    print("\n6. 处理输入...")
    try:
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": f"The item's text information: {text}",
                        },
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The item's text information: {text}",
                        }
                    ],
                }
            ]

        processed_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"处理后文本长度: {len(processed_text)}")

        images, _ = process_vision_info(messages)
        inputs = processor(
            text=processed_text,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)

        print("输入张量形状:")
        print(f"  input_ids: {inputs['input_ids'].shape}")
        print(f"  attention_mask: {inputs['attention_mask'].shape}")
        print(f"  有效token数: {inputs['attention_mask'].sum().item()}")

    except Exception as e:
        print(f"输入处理失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 7. 模型前向传播
    print("\n7. 模型前向传播...")
    try:
        with torch.no_grad():
            outputs = model(
                **inputs, output_hidden_states=True, return_dict=True
            )

        last_hidden_state = outputs.hidden_states[-1]
        print(f"隐藏状态形状: {last_hidden_state.shape}")
        print(f"隐藏状态数据类型: {last_hidden_state.dtype}")

        # 检查NaN和Inf
        nan_count = torch.isnan(last_hidden_state).sum().item()
        inf_count = torch.isinf(last_hidden_state).sum().item()

        print(f"NaN计数: {nan_count}")
        print(f"Inf计数: {inf_count}")

        if nan_count > 0:
            print("❌ 发现NaN！")
            # 找到NaN的位置
            nan_positions = torch.isnan(last_hidden_state).nonzero()
            print(f"NaN位置（前5个）: {nan_positions[:5]}")

        if inf_count > 0:
            print("❌ 发现Inf！")

        # 统计信息
        print(
            f"数值范围: [{last_hidden_state.min().item():.6f}, {last_hidden_state.max().item():.6f}]"
        )
        print(f"均值: {last_hidden_state.mean().item():.6f}")
        print(f"标准差: {last_hidden_state.std().item():.6f}")

    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 8. 计算最终表征
    print("\n8. 计算最终表征...")
    try:
        attention_mask = inputs["attention_mask"]
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        sum_embeddings = torch.sum(
            last_hidden_state * attention_mask_expanded, dim=1
        )
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

        print(f"分母最小值: {sum_mask.min().item()}")
        print(f"分母最大值: {sum_mask.max().item()}")

        mean_embeddings = sum_embeddings / sum_mask

        # 最终检查
        final_nan = torch.isnan(mean_embeddings).sum().item()
        final_inf = torch.isinf(mean_embeddings).sum().item()

        print(f"最终表征形状: {mean_embeddings.shape}")
        print(f"最终NaN计数: {final_nan}")
        print(f"最终Inf计数: {final_inf}")

        if final_nan == 0 and final_inf == 0:
            print("✅ 表征提取成功！")
            print(
                f"表征范围: [{mean_embeddings.min().item():.6f}, {mean_embeddings.max().item():.6f}]"
            )
        else:
            print("❌ 最终表征包含异常值！")

    except Exception as e:
        print(f"表征计算失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # 设置日志
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    debug_item_3186()
