#!/usr/bin/env python3
"""
Wrapper script to extract representations for multiple text modes.

This script is a light wrapper around data_process.qwen_embeddings.ItemMultimodalExtractor
and will generate one JSON+NPY pair per mode under data/<dataset>/reps/ by default.
"""

import argparse
import os

import torch

from data_process.qwen_embeddings import (
    ItemMultimodalBatchExtractor as ItemMultimodalExtractor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument("--model", type=str, default="Qwen2-VL-2B-Instruct")
    parser.add_argument("--out-dir", type=str, default="reps")
    parser.add_argument(
        "--modes", type=str, default="orig,orig_enhanced,enhanced"
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_path = os.path.abspath(os.path.join("data", args.dataset))
    out_dir = os.path.abspath(os.path.join("data", args.dataset, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # single-process: use first CUDA device if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # preserve optional model root prefix used in the original script
    root_dir = "/opt/meituan/dolphinfs_zhangkangning02/zkn/GRec/ckpt/base_model"
    model = os.path.join(root_dir, args.model)

    # build combinations: image_only (with images), and for each text mode with/without images
    text_modes = [m.strip() for m in args.modes.split(",")]
    combos = []
    # image_only (images included)
    combos.append(
        {
            "mode_tag": "image_only",
            "mode": "image_only",
            "include_image": True,
            "image_only": True,
        }
    )
    # for each text mode, produce with images and without images
    for tm in text_modes:
        combos.append(
            {
                "mode_tag": tm,
                "mode": tm,
                "include_image": True,
                "image_only": False,
            }
        )
        combos.append(
            {
                "mode_tag": tm,
                "mode": tm,
                "include_image": False,
                "image_only": False,
            }
        )

    fname_prefix = args.model.replace("/", "_")
    for combo in combos:
        mode_tag = combo["mode_tag"]
        img_flag = "img" if combo["include_image"] else "noimg"
        out_path = os.path.join(
            out_dir, f"{fname_prefix}_{mode_tag}_{img_flag}.json"
        )
        npy_path = out_path.replace(".json", ".npy")

        # skip if both json and npy exist
        if os.path.exists(out_path) and os.path.exists(npy_path):
            print(f"Skipping existing output: {out_path} (+ .npy)")
            continue

        print(
            f"Extracting mode={mode_tag} img={combo['include_image']} -> {out_path} on {device}"
        )

        extractor = ItemMultimodalExtractor(
            model_name=model,
            device=device,
            dataset=args.dataset,
            batch_size=8,
            mode=combo["mode"],
            include_image=combo["include_image"],
            image_only=combo.get("image_only", False),
        )

        # run will process all items and save output
        extractor.run(dataset_path=dataset_path, output_path=out_path)
    # no distributed synchronization or merging in single-process mode


if __name__ == "__main__":
    main()
