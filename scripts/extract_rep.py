#!/usr/bin/env python3
"""
Wrapper script to extract representations for multiple text modes.

This script is a light wrapper around data_process.qwen_embeddings.ItemMultimodalExtractor
and will generate multiple JSON+NPY pairs under `data/<dataset>/<out-dir>/` by default:

- `image_only` (with images if available)
- for each text mode: one run with images + one run text-only
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
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--model-root",
        type=str,
        default=None,
        help="Optional prefix directory for local checkpoints (joined with --model).",
    )
    parser.add_argument("--out-dir", type=str, default="reps")
    parser.add_argument(
        "--modes", type=str, default="orig,orig_enhanced,enhanced"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. cuda:0 / cpu. Default: auto-detect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N items (debug).",
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(os.path.join("data", args.dataset))
    out_dir = os.path.abspath(os.path.join("data", args.dataset, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # single-process: use first CUDA device if available
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    model = (
        os.path.join(args.model_root, args.model)
        if args.model_root
        else args.model
    )

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
        limit_tag = f"_limit{args.limit}" if args.limit else ""
        out_path = os.path.join(
            out_dir, f"{fname_prefix}_{mode_tag}_{img_flag}{limit_tag}.json"
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
            batch_size=args.batch_size,
            mode=combo["mode"],
            include_image=combo["include_image"],
            image_only=combo.get("image_only", False),
        )

        # run will process all items and save output
        extractor.run(
            dataset_path=dataset_path, output_path=out_path, limit=args.limit
        )
    # no distributed synchronization or merging in single-process mode


if __name__ == "__main__":
    main()
