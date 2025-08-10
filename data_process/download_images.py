import argparse
import json
import os

import requests
from tqdm import tqdm


def download_image(img_id, url_suffix, images_dir):
    url = f"http://ecx.images-amazon.com/images/I/{url_suffix}"
    target_path = os.path.join(images_dir, f"{img_id}.jpg")
    if os.path.exists(target_path):
        return True
    try:
        resp = requests.get(url, timeout=20, proxies=proxies)
        resp.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"Failed to download {img_id}: {url}, reason: {e}")
        return False


def main(dataset):
    dataset_path = os.path.abspath(os.path.join("data", dataset))
    images_info_path = os.path.abspath(
        os.path.join("data", "images_info", f"{dataset}_images_info.json")
    )
    images_dir = os.path.join(dataset_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    if not os.path.exists(images_info_path):
        print(f"images_info.json not found at: {images_info_path}")
        return

    with open(images_info_path, encoding="utf-8") as f:
        data = json.load(f)

    for img_id, url_list in tqdm(
        data.items(), desc=f"Downloading images for {dataset}"
    ):
        if isinstance(url_list, list) and len(url_list) > 0:
            url_suffix = url_list[0]
            download_image(img_id, url_suffix, images_dir)
        # 跳过空list或非list，不做处理


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images for dataset")
    parser.add_argument(
        "--dataset", type=str, default="Instrument", help="Dataset name"
    )
    # parser.add_argument('--proxy', type=str, default="http://127.0.0.1:8080", help='proxy, eg: http://127.0.0.1:8080')
    # args = parser.parse_args()
    # if args.proxy:
    proxies = {"http": "10.140.24.177:3128", "https": "10.140.15.68:3128"}
    for dataset in ["Instruments", "Arts", "Games"]:
        main(dataset)
