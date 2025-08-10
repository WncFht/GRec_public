import argparse
import json
import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


class ItemMultimodalExtractor:
    def __init__(
        self,
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dataset="Instruments",
    ) -> None:
        """
        初始化QwenVL-2.5模型用于多模态表征提取
        """
        self.device = device
        self.dataset = dataset
        print(f"Using device: {self.device}")

        # 加载模型和处理器
        print("Loading QwenVL-2.5 model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cuda"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model.eval()
        print("Model loaded successfully!")

    def load_dataset_info(
        self, dataset_path
    ) -> tuple[dict[str, int], dict[int, str], dict[str, Any]]:
        """
        加载数据集信息
        """
        # 加载item2id映射
        item2id_path = os.path.join(dataset_path, f"{self.dataset}.item2id")
        item2id = {}
        id2item = {}

        with open(item2id_path) as f:
            for line in f:
                item_id, num_id = line.strip().split("\t")
                item2id[item_id] = int(num_id)
                id2item[int(num_id)] = item_id

        # 加载物品信息
        item_info_path = os.path.join(dataset_path, f"{self.dataset}.item.json")
        with open(item_info_path) as f:
            item_info = json.load(f)

        return item2id, id2item, item_info

    def construct_item_text(self, item_data: dict[str, Any]) -> str:
        """
        构造物品的文本描述
        """
        text_parts = []

        if item_data.get("title"):
            text_parts.append(f"Title: {item_data['title']}")

        if item_data.get("brand"):
            text_parts.append(f"Brand: {item_data['brand']}")

        if item_data.get("categories"):
            text_parts.append(f"Categories: {item_data['categories']}")

        if item_data.get("description") and item_data["description"].strip():
            text_parts.append(f"Description: {item_data['description']}")

        # 添加enhanced信息（如果存在）
        if item_data.get("enhanced_title"):
            text_parts.append(f"Enhanced Title: {item_data['enhanced_title']}")

        if item_data.get("tags") and isinstance(item_data["tags"], list):
            text_parts.append(f"Tags: {', '.join(item_data['tags'])}")

        if item_data.get("highlights") and isinstance(
            item_data["highlights"], list
        ):
            text_parts.append(
                f"Highlights: {', '.join(item_data['highlights'])}"
            )

        if item_data.get("characteristics") and isinstance(
            item_data["characteristics"], list
        ):
            text_parts.append(
                f"Characteristics: {', '.join(item_data['characteristics'])}"
            )

        return " | ".join(text_parts)

    def load_item_image(self, image_path: str) -> Image.Image | None:
        """
        加载物品图片，如果不存在则返回None
        """
        if not os.path.exists(image_path):
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            return None

    def extract_item_representation(
        self, text: str, image: Image.Image | None = None
    ) -> np.ndarray | None:
        """
        提取单个物品的多模态表征
        """
        try:
            # 准备输入
            if image is not None:
                # 多模态输入
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
                # 纯文本输入
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

            # 处理输入
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, _ = process_vision_info(messages)
            inputs = self.processor(
                text=text, images=images, padding=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                )

            # 获取last hidden state
            last_hidden_state = outputs.hidden_states[
                -1
            ]  # [batch_size, seq_len, hidden_dim]

            if (
                torch.isnan(last_hidden_state).any()
                or torch.isinf(last_hidden_state).any()
            ):
                print(
                    "Warning: NaN or Inf detected in hidden states, skipping..."
                )
                return None

            # 计算有效token的平均 (排除padding token)
            attention_mask = inputs["attention_mask"]

            # 扩展attention_mask到hidden_dim
            attention_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(last_hidden_state.size())
                .float()
            )

            # 计算加权平均
            sum_embeddings = torch.sum(
                last_hidden_state * attention_mask_expanded, dim=1
            )
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            return mean_embeddings.cpu().numpy()

        except Exception as e:
            logging.exception(f"Error extracting representation: {e}")
            return None

    def extract_all_items(
        self,
        dataset_path: str,
        output_path: str,
        batch_size: int = 1,
    ) -> tuple[dict[int, dict[str, Any]], list[int]]:
        """
        提取所有物品的多模态表征
        """
        # 加载数据集信息
        print("Loading dataset information...")
        item2id, id2item, item_info = self.load_dataset_info(dataset_path)

        # case

        # 准备输出
        representations = {}
        failed_items = []

        # 获取图片目录
        image_dir = os.path.join(dataset_path, "images")

        print(f"Extracting representations for {len(item_info)} items...")

        # 遍历所有物品
        for num_id, item_data in tqdm(
            item_info.items(), desc="Processing items"
        ):
            num_id = int(num_id)
            item_id = id2item[num_id]

            # 构造文本
            text = self.construct_item_text(item_data)

            # 加载图片
            image_path = os.path.join(image_dir, f"{item_id}.jpg")
            # import pdb; pdb.set_trace()
            image = self.load_item_image(image_path)

            try:
                # 提取表征
                representation = self.extract_item_representation(text, image)

                if representation is not None:
                    print(
                        f"Successfully extract representation of item {num_id}"
                    )
                    representations[num_id] = {
                        "item_id": item_id,
                        "representation": representation.tolist(),
                        "has_image": image is not None,
                        "text": text,
                    }

                else:
                    print(f"The representaion of item {num_id} is None")
                    failed_items.append(num_id)

            except Exception as e:
                logging.exception(f"Failed to process item {num_id}: {e}")
                failed_items.append(num_id)
                continue

        # 保存结果
        print(f"Saving representations to {output_path}...")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(representations, f, indent=2)

        # 保存为numpy数组格式（更适合后续使用）
        numpy_output_path = output_path.replace(".json", ".npy")

        # 从 0 开始排列
        representation_matrix = [
            representations[num_id]["representation"]
            for num_id in sorted(representations.keys())
        ]

        representation_matrix = np.array(representation_matrix)
        np.save(numpy_output_path, representation_matrix)
        # np.save(numpy_output_path.replace('.npy', '_ids.npy'), np.array(item_ids))

        print(f"Successfully processed {len(representations)} items")
        print(f"Failed items: {len(failed_items)}")
        print(f"Representation shape: {representation_matrix.shape}")
        print(f"Results saved to: {output_path} and {numpy_output_path}")

        return representations, failed_items


def main(args):
    """
    主函数
    """
    # 配置参数
    dataset = args.dataset
    model = args.model
    dataset_path = os.path.abspath(os.path.join("data", dataset))
    output_path = os.path.abspath(
        os.path.join("data", dataset, f"{model}_rep.json")
    )

    # 创建提取器
    extractor = ItemMultimodalExtractor(model_name=model, dataset=dataset)

    # 提取表征
    representations, failed_items = extractor.extract_all_items(
        dataset_path=dataset_path, output_path=output_path, batch_size=1
    )

    print("Extraction completed!")


if __name__ == "__main__":
    # 设置日志
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    parser = argparse.ArgumentParser(description="Extract Multimodal Rep")
    parser.add_argument(
        "--dataset", type=str, default="Instruments", help="Dataset Name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Encoder Model",
    )
    args = parser.parse_args()
    main(args)
