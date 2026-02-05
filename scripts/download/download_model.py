import argparse
import os

from modelscope.hub.snapshot_download import snapshot_download


def download_qwen_model(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir: str = "../ckpt",
):
    """
    从ModelScope下载指定的Qwen模型。

    Args:
    ----
        model_name (str): 要下载的模型名称，例如 "qwen/Qwen2.5-VL-3B"。
        cache_dir (str): 模型下载后存放的本地缓存目录。

    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"创建缓存目录: {cache_dir}")

    print(f"正在从ModelScope下载模型: {model_name} 到 {cache_dir}...")
    try:
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        print(f"模型下载完成。模型路径: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"下载模型 {model_name} 失败: {e}")
        return None


if __name__ == "__main__":
    # input DATASET and BASE_MODEL by args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    args = parser.parse_args()
    BASE_MODEL = args.base_model
    OUTPUT_DIR = "./ckpt/base_model"
    # input cache_dir
    cache_dir = OUTPUT_DIR
    # 调用函数下载Qwen2.5-VL-3B模型
    downloaded_model_path = download_qwen_model(
        model_name=BASE_MODEL,
        cache_dir=cache_dir,
    )
    if downloaded_model_path:
        print(f"你可以在此路径找到下载的模型文件: {downloaded_model_path}")
    else:
        print("模型下载失败。请检查错误信息。")
