import argparse
import os

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer

from src.type import Args, TrainingSample
from src.utils import get_tokenizer

AR_MODEL = [
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen2",
    "qwen2_5",
    "llama",
    "llava_onevision",
    "qwen",
]


# Collator类，用于处理SFT（监督微调）任务的数据批次
class Collator:
    def __init__(self, args: Args, tokenizer):
        self.args = args
        # 是否只训练响应部分（即不计算instruction部分的损失）
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        # 如果tokenizer没有pad_token_id，则将其设置为unk_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # 确保 decoder-only 模型使用正确的 padding_side
        if hasattr(self.tokenizer, "padding_side"):
            # 对于decoder-only模型，使用left padding保持因果性
            if args.model_type in AR_MODEL:
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


class ChatTemplateCollator:
    """
    无图版 MultiModalCollator，mask 逻辑与 MultiModalCollator 完全一致
    """

    def __init__(self, args: Args, tokenizer: AutoTokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        if (
            getattr(self.tokenizer, "padding_side", None) != "left"
            and args.model_type in AR_MODEL
        ):
            self.tokenizer.padding_side = "left"

    def __call__(self, batch: list[TrainingSample]) -> dict[str, torch.Tensor]:
        # 1. 构造对话
        full_msgs, user_msgs = [], []
        for samp in batch:
            user = [{"role": "user", "content": samp.input_text}]
            assistant = [{"role": "assistant", "content": samp.label_text}]
            full_msgs.append(user + assistant)
            user_msgs.append(user)

        # 2. 渲染模板
        full_texts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in full_msgs
        ]
        user_texts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in user_msgs
        ]

        # 3. tokenize（两次，分别得到 full 和 user 的 ids）
        full_batch = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        user_batch = self.tokenizer(
            user_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        labels = full_batch["input_ids"].clone()

        # 4. mask 策略与 MultiModalCollator 完全一致
        if self.only_train_response:
            for i in range(len(batch)):
                # 有效 user token 长度（不含 pad）
                user_len = (
                    (user_batch["input_ids"][i] != self.tokenizer.pad_token_id)
                    .sum()
                    .item()
                )
                # 当前样本 pad 长度
                pad_len = (
                    (full_batch["input_ids"][i] == self.tokenizer.pad_token_id)
                    .sum()
                    .item()
                )
                # 掩掉 user + pad 部分
                if user_len > 0:
                    labels[i, : pad_len + user_len] = -100

        # 5. 再掩一次 pad（防止 pad 也被算 loss）
        labels[labels == self.tokenizer.pad_token_id] = -100

        # import pdb
        # pdb.set_trace()

        return {
            "input_ids": full_batch["input_ids"],
            "attention_mask": full_batch["attention_mask"],
            "labels": labels,
        }


class ChatTemplateTestCollator:
    """
    测试阶段使用的 Collator
    1. 只用 user 消息（不加 assistant 回复）
    2. 用 chat_template 渲染
    3. 返回 (inputs, targets, item_ids) 三元组，与 UnifiedTestCollator 保持一致
    """

    def __init__(self, args: Args, tokenizer: AutoTokenizer):
        self.args = args
        self.tokenizer = tokenizer

        # 补 pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # decoder-only 统一 left-pad
        if (
            getattr(self.tokenizer, "padding_side", None) != "left"
            and args.model_type in AR_MODEL
        ):
            self.tokenizer.padding_side = "left"

    def __call__(
        self, batch: list[TrainingSample]
    ) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
        # 1. 仅构造 user 消息
        user_messages = [
            [{"role": "user", "content": samp.input_text}] for samp in batch
        ]

        # 2. 用模板渲染（加 generation_prompt=True，与生成阶段一致）
        user_texts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in user_messages
        ]

        # 3. tokenize
        inputs = self.tokenizer(
            user_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True,
        )

        # 4. 组装与 UnifiedTestCollator 一致的返回格式
        targets = [samp.label_text for samp in batch]
        # print(batch[0])
        # print(type(inputs))
        # print(inputs)
        return (inputs, targets)


# MultiModalCollator类，用于处理多模态数据批次，支持混合批次
class MultiModalCollator:
    """多模态数据整理器 - 优化版本，支持混合batch"""

    def __init__(self, args: argparse, processor_or_tokenizer):
        self.args = args

        self.only_train_response = args.only_train_response
        self.tokenizer: AutoTokenizer = get_tokenizer(processor_or_tokenizer)
        self.processor = processor_or_tokenizer

        # 如果tokenizer没有pad_token_id，则将其设置为unk_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # 确保 decoder-only 模型使用正确的 padding_side
        if hasattr(self.tokenizer, "padding_side"):
            # 对于decoder-only模型，使用left padding保持因果性
            if args.model_type in AR_MODEL:
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
            # 构建用户消息内容，可以包含图像和文本
            user_content = []
            labels_content = []
            if item.image_path:
                # print("type(item.image_path):", type(item.image_path))
                if isinstance(item.image_path, str):
                    image_path_list = [item.image_path]
                else:
                    image_path_list = item.image_path

                # 检查是否有有效的图片路径，如果存在则添加到用户消息内容中
                for image_path in image_path_list:
                    if image_path and os.path.exists(image_path):
                        user_content.append(
                            {"type": "image", "image": image_path}
                        )

            # 添加文本内容
            user_content.append({"type": "text", "text": input_data})

            # 构建用户消息
            user_message = [{"role": "user", "content": user_content}]

            labels_content.append({"type": "text", "text": labels_data})

            # 完整对话
            full_messages = user_message + [
                {"role": "assistant", "content": labels_content}
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

                if user_len > 0:
                    labels[i, : pad_len + user_len] = -100

        # 掩码padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": batch_result["input_ids"],
            "attention_mask": batch_result["attention_mask"],
            "labels": labels,
        }


# UnifiedTestCollator类，统一的测试数据整理器
class UnifiedTestCollator:
    """统一的测试数据整理器 - 自动检测并处理单模态/多模态数据"""

    def __init__(
        self,
        args: argparse.Namespace,
        processor_or_tokenizer,
        model_type: str | None = None,
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
            if args.model_type in AR_MODEL:
                if self.tokenizer.padding_side != "left":
                    print(
                        f"调整 padding_side 从 {self.tokenizer.padding_side} 到 left，"
                        f"保持decoder-only模型的因果性"
                    )
                    self.tokenizer.padding_side = "left"

    def __call__(self, batch: list[TrainingSample]) -> tuple:
        """统一处理接口：根据batch内容自动选择单模态或多模态处理"""
        # 如果不支持多模态，或者batch中没有图像，则使用纯文本处理
        # if not self._has_images(batch):
        #     return self._process_text_only_batch(batch)
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
            user_content = []
            if item.image_path:
                # print("type(item.image_path):", type(item.image_path))
                if isinstance(item.image_path, str):
                    image_path_list = [item.image_path]
                else:
                    image_path_list = item.image_path

                # 检查是否有有效的图片路径，如果存在则添加到用户消息内容中
                for image_path in image_path_list:
                    if image_path and os.path.exists(image_path):
                        user_content.append(
                            {"type": "image", "image": image_path}
                        )

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

        if user_image_inputs or user_video_inputs:
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
        else:
            inputs = self.processor(
                text=input_texts,
                return_tensors="pt",  # 返回PyTorch tensors
                padding=True,  # 填充以确保批处理中所有序列长度一致
                truncation=True,  # 截断过长的序列
                max_length=self.tokenizer.model_max_length,  # 使用模型定义的最大长度
            )

        return (inputs, targets, item_ids)
