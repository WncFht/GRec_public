import argparse
import gc
import json
import logging
import os

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
    ):
        """
        初始化QwenVL-2.5模型用于多模态表征提取
        """
        self.device = device
        self.dataset = dataset
        print(f"Using device: {self.device}")

        # 加载模型和处理器
        print("Loading QwenVL-2.5 model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
            if device == "cuda"
            else torch.float32,  # 使用float16节省显存
            device_map="cuda" if device == "cuda" else "cpu",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model.eval()
        print("Model loaded successfully!")

    def load_dataset_info(self, dataset_path):
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

    def construct_item_text(self, item_data):
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

        return " | ".join(text_parts)

    def load_item_image(self, image_path):
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

    def prepare_batch_data(self, batch_items, image_dir):
        """
        准备批处理数据
        """
        batch_messages = []
        batch_info = []

        for num_id, item_data in batch_items:
            # 构造文本
            text = self.construct_item_text(item_data)

            # 加载图片
            image_path = os.path.join(
                image_dir, f"{item_data.get('item_id', '')}.jpg"
            )
            image = self.load_item_image(image_path)

            # 准备消息
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

            batch_messages.append(messages)
            batch_info.append(
                {"num_id": num_id, "text": text, "has_image": image is not None}
            )

        return batch_messages, batch_info

    def extract_batch_representations(self, batch_messages, batch_info):
        """
        批量提取多模态表征
        """
        try:
            # 准备批量输入
            texts = []
            all_images = []

            for messages in batch_messages:
                # 处理单个样本的文本
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                texts.append(text)

                # 处理图像
                images, _ = process_vision_info(messages)
                all_images.extend(images if images else [None])

            # 如果所有样本都没有图像，设置为None
            if all(img is None for img in all_images):
                all_images = None

            # 批量处理
            inputs = self.processor(
                text=texts, images=all_images, padding=True, return_tensors="pt"
            ).to(self.device)

            # 前向传播获取hidden states
            with torch.no_grad():
                outputs = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                )

            # 获取last hidden state
            last_hidden_state = outputs.hidden_states[
                -1
            ]  # [batch_size, seq_len, hidden_dim]

            # 检查NaN或Inf
            if (
                torch.isnan(last_hidden_state).any()
                or torch.isinf(last_hidden_state).any()
            ):
                print("Warning: NaN or Inf detected in hidden states")
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
            logging.exception(f"Error extracting batch representations: {e}")
            return None

    def extract_all_items_batch(self, dataset_path, output_path, batch_size=4):
        """
        批量提取所有物品的多模态表征
        """
        # 加载数据集信息
        print("Loading dataset information...")
        item2id, id2item, item_info = self.load_dataset_info(dataset_path)

        # 准备输出
        representations = {}
        failed_items = []

        # 获取图片目录
        image_dir = os.path.join(dataset_path, "images")

        print(
            f"Extracting representations for {len(item_info)} items with batch_size={batch_size}..."
        )

        # 准备所有物品数据
        all_items = []
        for num_id, item_data in item_info.items():
            num_id = int(num_id)
            item_id = id2item[num_id]
            item_data["item_id"] = item_id
            all_items.append((num_id, item_data))

        # 分批处理
        total_batches = (len(all_items) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(all_items), batch_size):
            print(f"-------Process batch {batch_idx}----------")
            batch_items = all_items[batch_idx : batch_idx + batch_size]

            try:
                # 准备批处理数据
                batch_messages, batch_info = self.prepare_batch_data(
                    batch_items, image_dir
                )

                # 提取批量表征
                batch_representations = self.extract_batch_representations(
                    batch_messages, batch_info
                )

                if batch_representations is not None:
                    # 保存结果
                    for i, info in enumerate(batch_info):
                        num_id = info["num_id"]
                        representations[num_id] = {
                            "item_id": id2item[num_id],
                            "representation": batch_representations[i].tolist(),
                            "has_image": info["has_image"],
                            "text": info["text"],
                        }
                    print(
                        f"Successfully processed batch {batch_idx // batch_size + 1}/{total_batches}"
                    )
                else:
                    # 如果批处理失败，尝试单独处理
                    print(f"Batch {batch_idx // batch_size + 1} failed")

                # 清理GPU内存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logging.exception(
                    f"Failed to process batch {batch_idx // batch_size + 1}: {e}"
                )
                for num_id, _ in batch_items:
                    failed_items.append(num_id)
                continue

        # 保存结果
        print(f"Saving representations to {output_path}...")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, "w") as f:
            json.dump(representations, f, indent=2)

        # 保存为numpy数组格式
        numpy_output_path = output_path.replace(".json", ".npy")
        representation_matrix = []

        for num_id in sorted(representations.keys()):
            representation_matrix.append(
                representations[num_id]["representation"]
            )

        if representation_matrix:
            representation_matrix = np.array(representation_matrix)
            np.save(numpy_output_path, representation_matrix)
            print(f"Representation shape: {representation_matrix.shape}")

        print(f"Successfully processed {len(representations)} items")
        print(f"Failed items: {len(failed_items)}")
        print(f"Results saved to: {output_path} and {numpy_output_path}")

        return representations, failed_items

    def extract_all_items(self, dataset_path, output_path, batch_size=1):
        """
        保持原有接口，根据batch_size选择处理方式
        """
        if batch_size > 1:
            return self.extract_all_items_batch(
                dataset_path, output_path, batch_size
            )
        # 原有的单个处理逻辑（保持向后兼容）
        return self.extract_all_items_single(dataset_path, output_path)

    def extract_all_items_single(self, dataset_path, output_path):
        """
        原有的单个物品处理逻辑（重构为单独方法）
        """
        # 加载数据集信息
        print("Loading dataset information...")
        item2id, id2item, item_info = self.load_dataset_info(dataset_path)

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
            image = self.load_item_image(image_path)

            try:
                # 提取表征
                representation = self.extract_item_representation_single(
                    text, image
                )

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
                    print(f"The representation of item {num_id} is None")
                    failed_items.append(num_id)

            except Exception as e:
                logging.exception(f"Failed to process item {num_id}: {e}")
                failed_items.append(num_id)
                continue

        # 保存结果的逻辑与批处理相同
        print(f"Saving representations to {output_path}...")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, "w") as f:
            json.dump(representations, f, indent=2)

        numpy_output_path = output_path.replace(".json", ".npy")
        representation_matrix = []

        for num_id in sorted(representations.keys()):
            representation_matrix.append(
                representations[num_id]["representation"]
            )

        if representation_matrix:
            representation_matrix = np.array(representation_matrix)
            np.save(numpy_output_path, representation_matrix)
            print(f"Representation shape: {representation_matrix.shape}")

        print(f"Successfully processed {len(representations)} items")
        print(f"Failed items: {len(failed_items)}")
        print(f"Results saved to: {output_path} and {numpy_output_path}")

        return representations, failed_items

    def extract_item_representation_single(self, text, image=None):
        """
        提取单个物品的多模态表征（原有逻辑）
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
            text_processed = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, _ = process_vision_info(messages)
            inputs = self.processor(
                text=text_processed,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # 前向传播获取hidden states
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


def main(args):
    """
    主函数
    """
    # 配置参数
    dataset = args.dataset
    model = args.model
    batch_size = args.batch_size
    dataset_path = os.path.abspath(os.path.join("data", dataset))
    output_path = os.path.abspath(
        os.path.join("data", dataset, f"{model.replace('/', '_')}_rep.json")
    )

    # 创建提取器
    extractor = ItemMultimodalExtractor(model_name=model, dataset=dataset)

    # 提取表征
    representations, failed_items = extractor.extract_all_items(
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=batch_size,
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
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Encoder Model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    args = parser.parse_args()

    main(args)
