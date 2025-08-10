# src/inspect_embeddings.py
import argparse
import os

import torch


def inspect_embeddings(file_path: str) -> None:
    """
    加载一个 .pt 文件并检查其内容。

    Args:
    ----
        file_path (str): .pt 文件的路径。

    """
    print(f"--- 正在检查文件: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在于 '{file_path}'")
        return

    try:
        # 加载文件，使用 map_location='cpu' 避免加载到特定的GPU
        data = torch.load(file_path, map_location="cpu")

        if isinstance(data, dict):
            print("文件包含一个字典，包含以下键:")
            for key, tensor in data.items():
                print(f"\n[键: '{key}']")
                if isinstance(tensor, torch.Tensor):
                    print(f"  - 形状 (Shape): {tensor.shape}")
                    print(f"  - 数据类型 (Dtype): {tensor.dtype}")
                    print(f"  - 设备 (Device): {tensor.device}")
                else:
                    print(f"  - 类型: {type(tensor)}")
                    print(f"  - 值: {tensor}")
        elif isinstance(data, torch.Tensor):
            print("文件包含一个单独的张量:")
            print(f"  - 形状 (Shape): {data.shape}")
            print(f"  - 数据类型 (Dtype): {data.dtype}")
            print(f"  - 设备 (Device): {data.device}")
        else:
            print(f"文件包含的数据类型: {type(data)}")
            print(f"值: {data}")

        print("\n--- 检查完成 ---")

    except Exception as e:
        print(f"检查文件时发生错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 .pt embedding 文件。")
    parser.add_argument(
        "file_path",
        type=str,
        help="要检查的 new_token_embeddings.pt 文件的路径。",
    )
    args = parser.parse_args()
    inspect_embeddings(args.file_path)
