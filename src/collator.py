import copy
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
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch: list[TrainingSample]):
        # 从批次中提取输入文本和完整文本（包括标签和EOS token）
        input_texts = [d.input_text for d in batch]
        # 在SFT任务中，标签通常是完整的 instruction + response
        full_texts = [d.label_text + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text=full_texts,
            # text_target=input_texts, # 移除 text_target，因为它在此处用途不正确
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])  # labels 是 tokenized full_texts
        if self.only_train_response:
            # 忽略 padding token
            labels[labels == self.tokenizer.pad_token_id] = -100

            # 重新 tokenized input_texts 以获取其真实长度，用于掩码指令部分
            instruction_tokenized_inputs = self.tokenizer(
                text=input_texts,
                return_tensors="pt",
                padding="longest",  # 对齐子批次中的长度
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )

            # 逐样本掩码指令部分
            for i in range(len(batch)):
                # 获取当前样本指令的实际非填充长度
                instruction_len = (
                    instruction_tokenized_inputs["input_ids"][i]
                    != self.tokenizer.pad_token_id
                ).sum().item()

                # 掩码指令部分，将其标签设置为 -100
                if instruction_len > 0:
                    labels[i, :instruction_len] = -100

        inputs["labels"] = labels

        return inputs


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

    def __call__(self, batch: list[TrainingSample]) -> dict[str, torch.Tensor]:
        """处理一批多模态数据 - 支持混合batch"""
        # Text-only logic for models like Llama
        # Check if the batch is text-only by looking at the first sample
        if not batch or not batch[0].is_multimodal:
            input_texts = [d.input_text for d in batch]
            # 兼容SFT训练，label_text可能已经是完整的 instruction+response
            # 在纯文本模式下，我们假设label_text是完整的
            full_texts = [d.label_text + self.tokenizer.eos_token for d in batch]

            inputs = self.tokenizer(
                text=full_texts,
                # text_target=input_texts, # 移除 text_target
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )
            labels = copy.deepcopy(inputs["input_ids"])  # labels 是 tokenized full_texts
            if self.only_train_response:
                # 忽略 padding token
                labels[labels == self.tokenizer.pad_token_id] = -100

                # 重新 tokenized input_texts 以获取其真实长度，用于掩码指令部分
                instruction_tokenized_inputs = self.tokenizer(
                    text=input_texts,
                    return_tensors="pt",
                    padding="longest",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                )

                # 逐样本掩码指令部分
                for i in range(len(batch)):
                    # 获取当前样本指令的实际非填充长度
                    instruction_len = (
                        instruction_tokenized_inputs["input_ids"][i]
                        != self.tokenizer.pad_token_id
                    ).sum().item()
                    # 掩码指令部分，将其标签设置为 -100
                    if instruction_len > 0:
                        labels[i, :instruction_len] = -100

            inputs["labels"] = labels
            return inputs

        # Existing multimodal logic
        # 分别构造用户消息和完整对话列表
        user_messages_list = []
        full_messages_list = []

        for item in batch:
            input_data = item.input_text
            labels_data = item.label_text
            image_path = (
                item.image_path
                if item.image_path
                else ""
            )

            # 构建用户消息内容，支持图像和文本
            user_content = []

            # 检查是否有有效的图片路径，如果存在则添加到用户消息内容中
            if image_path and os.path.exists(image_path):
                user_content.append({"type": "image", "image": image_path})

            # 添加文本内容到用户消息内容中
            user_content.append({"type": "text", "text": input_data})

            # 构建用户消息（遵循聊天模板的角色格式）
            user_message = [{"role": "user", "content": user_content}]

            # 构建完整对话（用户消息 + 助手响应）
            full_messages = [
                *user_message,
                {"role": "assistant", "content": labels_data},
            ]

            # 将构建好的消息添加到各自的列表中
            user_messages_list.append(user_message)
            full_messages_list.append(full_messages)

        # 统一提取所有对话的视觉信息（图像和视频）
        all_image_inputs, all_video_inputs = process_vision_info(
            full_messages_list
        )

        # 批量应用聊天模板到完整对话文本，不进行tokenize，不添加生成提示
        full_texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in full_messages_list
        ]

        # 批量处理完整对话，包括文本和视觉输入
        batch_result = self.processor(
            text=full_texts,
            images=all_image_inputs,  # 统一处理的图像输入
            videos=all_video_inputs,  # 视频输入（如果有）
            return_tensors="pt",  # 返回PyTorch tensors
            padding=True,  # 填充序列
        )

        # 创建标签：克隆input_ids作为初始标签
        labels = batch_result["input_ids"].clone()

        # 处理only_train_response逻辑：如果只训练响应部分
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

            # 批量tokenize用户输入来计算长度，用于确定要忽略的标签部分
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

                # 掩码用户输入部分，将其标签设置为-100，以便在损失计算中忽略
                if user_len > 0:
                    labels[i, :user_len] = -100

        # 掩码padding tokens，将其标签设置为-100
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 将处理后的标签添加到batch_result中
        batch_result["labels"] = labels

        return batch_result


# UnifiedTestCollator类，统一的测试数据整理器
class UnifiedTestCollator:
    """统一的测试数据整理器 - 自动检测并处理单模态/多模态数据"""

    def __init__(self, args: Args, processor_or_tokenizer, model_type: str | None = None):
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

    def _process_text_only_batch(
        self, batch: list[TrainingSample]
    ) -> tuple:
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

    def _process_multimodal_batch(
        self, batch: list[TrainingSample]
    ) -> tuple:
        """处理多模态batch"""
        messages_list = []  # 存储所有样本的用户消息列表
        targets = []  # 存储所有样本的目标文本列表
        item_ids = [] # 存储item_id

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

        # 统一提取所有对话的视觉信息（图像和视频）
        all_image_inputs, all_video_inputs = process_vision_info(messages_list)

        if self.model_type == "instructblip":
            input_texts = [item.input_text for item in batch]
        else:
            # 批量应用聊天模板到输入文本，不进行tokenize，添加生成提示（因为是测试阶段）
            input_texts = [
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in messages_list
            ]

        # 批量处理输入，包括文本、图像、视频
        inputs = self.processor(
            text=input_texts,
            images=all_image_inputs,
            videos=all_video_inputs,
            return_tensors="pt",  # 返回PyTorch tensors
            padding=True,  # 填充以确保批处理中所有序列长度一致
            truncation=True,  # 截断过长的序列
            max_length=self.tokenizer.model_max_length,  # 使用模型定义的最大长度
        )

        return (inputs, targets, item_ids)
