import base64
import os
from typing import Any

from openai import OpenAI


class MultiModalGPT:
    def __init__(self):
        self.client = OpenAI(
            api_key="761cae0b42c4444da750e0d3129dbe39",
            base_url="https://runway.devops.xiaohongshu.com/openai",
            default_headers={"api-key": "761cae0b42c4444da750e0d3129dbe39"},
        )

    def encode_image(self, image_path: str) -> str:
        """将图片文件编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_image_message(
        self, image_path: str, text_prompt: str
    ) -> list[dict[str, Any]]:
        """创建包含图片和文本的消息"""
        base64_image = self.encode_image(image_path)

        return [
            {
                "role": "system",
                "content": "你是一个能够分析图片和回答问题的AI助手。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",  # 可选: "low", "high", "auto"
                        },
                    },
                ],
            },
        ]

    def create_multiple_images_message(
        self, image_paths: list[str], text_prompt: str
    ) -> list[dict[str, Any]]:
        """创建包含多张图片和文本的消息"""
        content = [{"type": "text", "text": text_prompt}]

        for image_path in image_paths:
            base64_image = self.encode_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    },
                }
            )

        return [
            {
                "role": "system",
                "content": "你是一个能够分析多张图片并回答问题的AI助手。",
            },
            {"role": "user", "content": content},
        ]

    def get_multimodal_response(self, messages: list[dict[str, Any]]) -> str:
        """获取多模态响应"""
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4-vision-preview",  # 或者使用支持vision的其他模型
                messages=messages,
                stream=False,
                max_tokens=1000,  # 增加token限制以获得更详细的回答
                temperature=0.7,
                extra_query={"api-version": "2024-12-01-preview"},
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"API调用失败: {e!s}"


# 使用示例
def main():
    gpt = MultiModalGPT()

    # 示例1: 单张图片分析
    image_path = "data/Arts/images/8862933177.jpg"
    if os.path.exists(image_path):
        messages = gpt.create_image_message(
            image_path=image_path,
            text_prompt="请详细描述这张图片中的内容，包括主要物体、颜色、场景等。",
        )

        response = gpt.get_multimodal_response(messages)
        print("单张图片分析结果:")
        print(response)
        print("-" * 50)

    # 示例2: 多张图片对比分析
    image_paths = [
        "data/Instruments/images/0739079891.jpg",
        "data/Instruments/images/0786615206.jpg",
    ]
    if all(os.path.exists(path) for path in image_paths):
        messages = gpt.create_multiple_images_message(
            image_paths=image_paths,
            text_prompt="请对比分析这些图片的异同点，并总结它们的共同特征。",
        )

        response = gpt.get_multimodal_response(messages)
        print("多张图片对比分析:")
        print(response)


if __name__ == "__main__":
    main()
