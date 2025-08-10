import argparse
import json
import logging
import os

import numpy as np
import torch
from PIL import Image
from src.arguments import ModelArguments
from src.model.model import MMEBModel
from src.model.processor import QWEN2_VL, VLM_IMAGE_TOKENS, load_processor
from tqdm import tqdm

# 设置全局日志级别为ERROR，只显示错误信息
logging.basicConfig(level=logging.ERROR)


class ItemVLM2VecExtractor:
    def __init__(
        self,
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        checkpoint_path="TIGER-Lab/VLM2Vec-Qwen2VL-7B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dataset="Instruments",
    ):
        """
        初始化VLM2Vec模型用于多模态表征提取
        """
        self.device = device
        self.dataset = dataset
        print(f"Using device: {self.device}")

        # 设置VLM2Vec模型参数
        print("Loading VLM2Vec model...")
        self.model_args = ModelArguments(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            pooling="last",
            normalize=True,
            model_backbone="qwen2_vl",
            lora=True,
        )

        # 加载processor和model
        self.processor = load_processor(self.model_args)
        self.processor = self.fix_processor_config(self.processor)
        self.model = MMEBModel.load(self.model_args)
        self.model = self.model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()

        print("VLM2Vec model loaded successfully!")

    def fix_processor_config(self, processor):
        """修复处理器中的None值配置"""
        # 设置图像处理器的关键参数
        if processor.image_processor.max_pixels is None:
            processor.image_processor.max_pixels = 1280 * 28 * 28  # 约100万像素

        if processor.image_processor.min_pixels is None:
            processor.image_processor.min_pixels = 4 * 28 * 28  # 约3136像素

        # 设置尺寸参数
        if processor.image_processor.size.get("longest_edge") is None:
            processor.image_processor.size["longest_edge"] = 1024

        if processor.image_processor.size.get("shortest_edge") is None:
            processor.image_processor.size["shortest_edge"] = 512

        return processor

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

    def load_item_image(self, image_path):
        """
        加载物品图片，如果不存在则返回None
        """
        if not os.path.exists(image_path):
            return None

        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            return None

    def extract_item_representation(self, text, image=None):
        """
        使用VLM2Vec提取单个物品的多模态表征
        """
        # try:
        if image is not None:
            # 多模态输入 - 使用图像+文本
            prompt_text = f"{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given item with the following information: {text}"
            inputs = self.processor(
                text=prompt_text, images=image, return_tensors="pt"
            )
        else:
            # 纯文本输入
            prompt_text = f"Represent the given item with the following information: {text}"
            inputs = self.processor(
                text=prompt_text, images=None, return_tensors="pt"
            )

        # 将输入移动到设备
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
        inputs["image_grid_thw"] = inputs["image_grid_thw"].unsqueeze(0)
        # import pdb; pdb.set_trace()
        # 处理图像相关的输入维度
        # if image is not None and 'pixel_values' in inputs:
        #     if len(inputs['pixel_values'].shape) == 3:
        #         inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        #     if 'image_grid_thw' in inputs and len(inputs['image_grid_thw'].shape) == 1:
        #         inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)

        with torch.no_grad():
            # 使用qry模式提取表征
            outputs = self.model(qry=inputs)
            representation = outputs["qry_reps"]

            if (
                torch.isnan(representation).any()
                or torch.isinf(representation).any()
            ):
                print(
                    "Warning: NaN or Inf detected in representations, skipping..."
                )
                return None

            return representation.cpu().float().numpy()

        # except Exception as e:
        #     logging.error(f"Error extracting representation: {e}")
        #     return None

    def extract_all_items(self, dataset_path, output_path, batch_size=1):
        """
        提取所有物品的多模态表征
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
            f"Extracting VLM2Vec representations for {len(item_info)} items..."
        )

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

            # try:
            # 提取表征
            representation = self.extract_item_representation(text, image)

            if representation is not None:
                print(
                    f"Successfully extract VLM2Vec representation of item {num_id}"
                )
                representations[num_id] = {
                    "item_id": item_id,
                    "representation": representation.tolist(),
                    "has_image": image is not None,
                    "text": text,
                }
            else:
                print(f"The VLM2Vec representation of item {num_id} is None")
                failed_items.append(num_id)

            # except Exception as e:
            #     logging.error(f"Failed to process item {num_id}: {e}")
            #     failed_items.append(num_id)
            #     continue

        # 保存结果
        print(f"Saving VLM2Vec representations to {output_path}...")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, "w") as f:
            json.dump(representations, f, indent=2)

        # 保存为numpy数组格式（更适合后续使用）
        numpy_output_path = output_path.replace(".json", ".npy")
        representation_matrix = []

        for num_id in sorted(representations.keys()):  # 从0开始排列
            representation_matrix.append(
                representations[num_id]["representation"]
            )

        representation_matrix = np.array(representation_matrix)
        np.save(numpy_output_path, representation_matrix)

        print(f"Successfully processed {len(representations)} items")
        print(f"Failed items: {len(failed_items)}")
        print(f"VLM2Vec representation shape: {representation_matrix.shape}")
        print(f"Results saved to: {output_path} and {numpy_output_path}")

        return representations, failed_items


def main(args):
    """
    主函数
    """
    # 配置参数
    dataset = args.dataset
    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    dataset_path = os.path.abspath(os.path.join("data", dataset))
    output_path = os.path.abspath(
        os.path.join("data", dataset, "VLM2Vec_rep.json")
    )

    # 创建提取器
    extractor = ItemVLM2VecExtractor(
        model_name=model_name, checkpoint_path=checkpoint_path, dataset=dataset
    )

    # 提取表征
    representations, failed_items = extractor.extract_all_items(
        dataset_path=dataset_path, output_path=output_path, batch_size=1
    )

    print("VLM2Vec extraction completed!")


if __name__ == "__main__":
    # 设置环境变量
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"

    parser = argparse.ArgumentParser(
        description="Extract VLM2Vec Multimodal Representations"
    )
    parser.add_argument(
        "--dataset", type=str, default="Instruments", help="Dataset Name"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Base Model Name",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="TIGER-Lab/VLM2Vec-Qwen2VL-7B",
        help="VLM2Vec Checkpoint Path",
    )
    args = parser.parse_args()

    main(args)
