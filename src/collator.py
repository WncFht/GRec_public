import os
from typing import TYPE_CHECKING

import torch
from qwen_vl_utils import process_vision_info

from .type import Args, TrainingSample
from .utils import get_tokenizer

if TYPE_CHECKING:
    from transformers import AutoTokenizer


# Collator类，用于处理SFT（监督微调）任务的数据批次
class Collator:
    def __init__(self, args: Args, tokenizer):
        self.args = args
        # 是否只训练响应部分（即不计算instruction部分的损失）
        self.only_train_response = args.dataset_args.only_train_response
        self.tokenizer = tokenizer
        # 如果tokenizer没有pad_token_id，则将其设置为unk_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # 确保 decoder-only 模型使用正确的 padding_side
        if hasattr(self.tokenizer, "padding_side"):
            # 对于decoder-only模型，使用left padding保持因果性
            if args.global_args.model_type in ["qwen_vl", "llama", "qwen"]:
                if self.tokenizer.padding_side != "left":
                    print(
                        f"调整 padding_side 从 {self.tokenizer.padding_side} 到 left，"
                        f"保持decoder-only模型的因果性"
                    )
                    self.tokenizer.padding_side = "left"
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch: list[TrainingSample]):
        # 从批次中提取输入文本和完整文本（包括标签和EOS token）
        input_texts = [d.input_text for d in batch]
        full_texts = [
            d.input_text + d.label_text + self.tokenizer.eos_token
            for d in batch
        ]

        # 使用tokenizer对完整文本进行编码，并生成目标文本（用于计算损失）
        inputs = self.tokenizer(
            text=full_texts,  # 模型的输入，通常是完整的对话（instruction + response）
            text_target=input_texts,  # 用于生成labels的文本，这里是input_texts
            return_tensors="pt",  # 返回PyTorch tensors
            padding="longest",  # 填充到批次中最长的序列
            max_length=self.tokenizer.model_max_length,  # 最大序列长度
            truncation=True,  # 截断超过最大长度的序列
            return_attention_mask=True,  # 返回attention mask
        )
        # 复制input_ids作为初始标签
        labels = inputs["input_ids"].clone()
        # 如果只训练响应部分
        if self.only_train_response:
            # 忽略padding token的损失，将其标签设置为-100
            labels[labels == self.tokenizer.pad_token_id] = -100
            # 忽略输入文本部分的损失，将其标签设置为-100
            labels[
                torch.where(
                    inputs["labels"] != self.tokenizer.pad_token_id
                )  # 这里inputs["labels"]实际上是inputs["text_target_ids"]
            ] = -100

        # 将处理后的标签赋值给inputs字典
        inputs["labels"] = labels

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }


class TestCollator:
    def __init__(self, args: Args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        # if isinstance(self.tokenizer, LlamaTokenizer):
        #     # Allow batched inference
        #     self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        # 从批次中提取输入文本和目标文本
        input_texts = [d.input_text for d in batch]
        targets = [d.label_text for d in batch]
        # 使用tokenizer处理输入文本
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)


# MultiModalCollator类，用于处理多模态数据批次，支持混合批次
class MultiModalCollator:
    """多模态数据整理器 - 优化版本，支持混合batch"""

    def __init__(self, args: Args, processor_or_tokenizer):
        self.args = args

        self.only_train_response = args.dataset_args.only_train_response
        self.tokenizer: AutoTokenizer = get_tokenizer(processor_or_tokenizer)
        self.processor = processor_or_tokenizer

        # 如果tokenizer没有pad_token_id，则将其设置为unk_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # 确保 decoder-only 模型使用正确的 padding_side
        if hasattr(self.tokenizer, "padding_side"):
            # 对于decoder-only模型，使用left padding保持因果性
            if args.global_args.model_type in ["qwen_vl", "llama", "qwen"]:
                if self.tokenizer.padding_side != "left":
                    print(
                        f"调整 padding_side 从 {self.tokenizer.padding_side} 到 left，"
                        f"保持decoder-only模型的因果性"
                    )
                    self.tokenizer.padding_side = "left"

    def __call__(self, batch: list[TrainingSample]) -> dict[str, torch.Tensor]:
        """处理一批多模态数据 - 支持混合batch"""
        # 分别构造用户消息和完整对话
        user_messages_list = []
        full_messages_list = []

        for item in batch:
            input_data = item.input_text
            labels_data = item.label_text
            image_path = item.image_path

            # 构建用户消息内容
            user_content = []

            # 检查是否有有效的图片路径
            if image_path and os.path.exists(image_path):
                user_content.append({"type": "image", "image": image_path})

            # 添加文本内容
            user_content.append({"type": "text", "text": input_data})

            # 构建用户消息
            user_message = [{"role": "user", "content": user_content}]

            # 完整对话
            full_messages = user_message + [
                {"role": "assistant", "content": labels_data}
            ]

            user_messages_list.append(user_message)
            full_messages_list.append(full_messages)

        # 统一提取所有对话的视觉信息
        all_image_inputs, all_video_inputs = process_vision_info(
            full_messages_list
        )

        # 批量应用聊天模板
        full_texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in full_messages_list
        ]

        # 批量处理完整对话
        batch_result = self.processor(
            text=full_texts,
            images=all_image_inputs,  # 统一处理的图像输入
            videos=all_video_inputs,  # 视频输入（如果有）
            return_tensors="pt",
            padding=True,
        )

        # 创建标签
        labels = batch_result["input_ids"].clone()

        # 处理only_train_response逻辑
        if self.only_train_response:
            # 批量计算用户输入长度
            user_texts = [
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in user_messages_list
            ]

            # 提取用户消息的视觉信息
            user_image_inputs, user_video_inputs = process_vision_info(
                user_messages_list
            )

            # 批量tokenize用户输入来计算长度
            user_batch_result = self.processor(
                text=user_texts,
                images=user_image_inputs,
                videos=user_video_inputs,
                return_tensors="pt",
                padding=True,
            )

            # 为每个样本设置标签掩码
            for i in range(len(batch)):
                # 找到用户输入的实际长度（排除padding）
                user_input_ids = user_batch_result["input_ids"][i]
                user_len = (
                    (user_input_ids != self.tokenizer.pad_token_id).sum().item()
                )
                pad_len = (
                    (
                        batch_result["input_ids"][i]
                        == self.tokenizer.pad_token_id
                    )
                    .sum()
                    .item()
                )

                # 这里有大问题，因为labels是 batch_result["input_ids"]，所以labels[i, :user_len] = -100 会导致有剩下的没覆盖掉
                # 掩码用户输入部分
                if user_len > 0:
                    labels[i, : pad_len + user_len] = -100

        # 掩码padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": user_batch_result["input_ids"],
            "attention_mask": batch_result["attention_mask"],
            "labels": labels,
        }


# UnifiedTestCollator类，统一的测试数据整理器
class UnifiedTestCollator:
    """统一的测试数据整理器 - 自动检测并处理单模态/多模态数据"""

    def __init__(
        self, args: Args, processor_or_tokenizer, model_type: str | None = None
    ):
        self.args = args
        self.model_type = model_type

        # 根据传入的参数，可以是processor（多模态）或tokenizer（单模态）
        self.processor = processor_or_tokenizer
        # 兼容 processor 和 tokenizer 两种情况
        if hasattr(processor_or_tokenizer, "tokenizer"):
            self.tokenizer = processor_or_tokenizer.tokenizer
        else:
            self.tokenizer = processor_or_tokenizer

        # 如果tokenizer没有pad_token_id，则使用unk_token_id或0作为默认值
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = getattr(
                self.tokenizer, "unk_token_id", 0
            )

        # 确保 decoder-only 模型使用正确的 padding_side
        if hasattr(self.tokenizer, "padding_side"):
            # 对于decoder-only模型，使用left padding保持因果性
            if self.model_type in ["qwen_vl", "llama", "qwen"]:
                if self.tokenizer.padding_side != "left":
                    print(
                        f"调整 padding_side 从 {self.tokenizer.padding_side} 到 left，"
                        f"保持decoder-only模型的因果性"
                    )
                    self.tokenizer.padding_side = "left"

    def __call__(self, batch: list[TrainingSample]) -> tuple:
        """统一处理接口：根据batch内容自动选择单模态或多模态处理"""
        # 如果不支持多模态，或者batch中没有图像，则使用纯文本处理
        if not self._has_images(batch):
            return self._process_text_only_batch(batch)
        # 否则使用多模态处理
        return self._process_multimodal_batch(batch)

    def _has_images(self, batch: list[TrainingSample]) -> bool:
        """检查batch中是否包含有效图像"""
        # 遍历batch中的每个item，检查is_multimodal标志和image_path
        return any(
            item.is_multimodal
            and item.image_path
            and os.path.exists(item.image_path)
            for item in batch
        )

    def _process_text_only_batch(self, batch: list[TrainingSample]) -> tuple:
        """处理纯文本batch"""
        # 提取输入文本和目标文本
        input_texts = [item.input_text for item in batch]
        targets = [item.label_text for item in batch]

        # 使用tokenizer处理文本输入
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",  # 返回PyTorch tensors
            padding="longest",  # 填充到batch中最长的序列
            truncation=True,  # 截断超过最大长度的序列
            return_attention_mask=True,  # 返回attention mask
        )

        return (inputs, targets)

    def _process_multimodal_batch(self, batch: list[TrainingSample]) -> tuple:
        """处理多模态batch"""
        messages_list = []  # 存储所有样本的用户消息列表
        targets = []  # 存储所有样本的目标文本列表
        item_ids = []  # 存储item_id

        for item in batch:
            image_path = item.image_path

            # 构建用户消息内容，可以包含图像和文本
            user_content = []

            # 检查是否有有效的图片路径，如果存在则添加到用户消息内容中
            if image_path and os.path.exists(image_path):
                user_content.append({"type": "image", "image": image_path})

            # 添加文本内容到用户消息内容中
            user_content.append({"type": "text", "text": item.input_text})

            # 构建用户消息（遵循聊天模板的角色格式）
            user_message = [{"role": "user", "content": user_content}]

            # 将构建好的用户消息和目标文本添加到各自的列表中
            messages_list.append(user_message)
            targets.append(item.label_text)
            if item.item_id:
                item_ids.append(item.item_id)

        # 提取用户消息的视觉信息（图像和视频）
        user_image_inputs, user_video_inputs = process_vision_info(
            messages_list
        )

        # 应用聊天模板到用户输入文本，添加生成提示
        input_texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_list
        ]

        # 批量处理用户输入，包括文本、图像、视频
        inputs = self.processor(
            text=input_texts,
            images=user_image_inputs,
            videos=user_video_inputs,
            return_tensors="pt",  # 返回PyTorch tensors
            padding=True,  # 填充以确保批处理中所有序列长度一致
            truncation=True,  # 截断过长的序列
            max_length=self.tokenizer.model_max_length,  # 使用模型定义的最大长度
        )

        return (inputs, targets, item_ids)
