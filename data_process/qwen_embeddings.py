import argparse
import gc
import json
import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Abstract lightweight wrapper that standardizes preprocess/forward/extract for different VL models.

    For now we implement simple wrappers for Qwen2.5-VL and Qwen2-VL. The wrapper exposes:
      - processor: the AutoProcessor (or equivalent) for input construction
      - model: the HF model object
      - preprocess(messages) -> inputs (device tensors)
      - forward(inputs) -> raw outputs
      - extract_embeddings(raw_outputs, inputs) -> np.ndarray(B, D)
    """

    @classmethod
    def from_pretrained(cls, model_name: str, device: str, torch_dtype):
        # Simple selection: if name contains '2.5' or '2.5-VL' prefer Qwen2.5 wrapper;
        # if contains 'Qwen2' or '2-VL' use Qwen2 wrapper. Default to Qwen25Wrapper.
        name_lower = model_name.lower()
        # Llava detection
        if "llava" in name_lower:
            return LlavaOnevisionWrapper.from_pretrained(
                model_name, device, torch_dtype
            )
        if "2.5" in name_lower or "2.5-vl" in name_lower:
            return Qwen25Wrapper.from_pretrained(
                model_name, device, torch_dtype
            )
        # fallback to Qwen2 wrapper (they share API in our current env)
        return Qwen2Wrapper.from_pretrained(model_name, device, torch_dtype)


class Qwen25Wrapper(ModelWrapper):
    @classmethod
    def from_pretrained(cls, model_name: str, device: str, torch_dtype):
        # reuse Qwen2_5_VLForConditionalGeneration and AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map={"": device}
        )
        processor = AutoProcessor.from_pretrained(model_name)
        inst = cls()
        inst.model = model
        inst.processor = processor
        inst.device = device
        return inst

    def preprocess(self, batch_messages: list):
        # build text and images similar to previous code
        texts = []
        images_all = []
        for messages in batch_messages:
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            images, _ = process_vision_info(messages)
            images_all.append(images if images else None)

        imgs = None if all(im is None for im in images_all) else images_all
        inputs = self.processor(
            text=texts, images=imgs, padding=True, return_tensors="pt"
        ).to(self.device)
        return inputs

    def forward(self, inputs):
        return self.model(**inputs, output_hidden_states=True, return_dict=True)

    def extract_embeddings(self, raw_outputs, inputs):
        last_hidden_state = raw_outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(
            last_hidden_state * attention_mask_expanded, dim=1
        )
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings.cpu().numpy()


class Qwen2Wrapper(Qwen25Wrapper):
    # For now Qwen2 shares the same implementation as Qwen2.5 in this repo.
    @classmethod
    def from_pretrained(cls, model_name: str, device: str, torch_dtype):
        # reuse same class loader - if there is a specific class for Qwen2 replace here
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map={"": device}
        )
        processor = AutoProcessor.from_pretrained(model_name)
        inst = cls()
        inst.model = model
        inst.processor = processor
        inst.device = device
        return inst


class LlavaOnevisionWrapper(ModelWrapper):
    @classmethod
    def from_pretrained(cls, model_name: str, device: str, torch_dtype):
        # Use Hugging Face transformers' LlavaOnevisionForConditionalGeneration
        # (requires a transformers build / package that exposes this class)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map={"": device}
        )
        processor = AutoProcessor.from_pretrained(model_name)
        inst = cls()
        inst.model = model
        inst.processor = processor
        inst.device = device
        return inst

    def preprocess(self, batch_messages: list):
        texts = []
        images_all = []
        for messages in batch_messages:
            if hasattr(self.processor, "apply_chat_template"):
                texts.append(
                    self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
            else:
                txt_parts = []
                for m in messages:
                    for c in m.get("content", []):
                        if c.get("type") == "text":
                            txt_parts.append(c.get("text", ""))
                texts.append(" \n ".join(txt_parts))

            images, _ = process_vision_info(messages)
            images_all.append(images if images else None)

        imgs = None if all(im is None for im in images_all) else images_all
        inputs = self.processor(
            text=texts, images=imgs, padding=True, return_tensors="pt"
        ).to(self.device)
        return inputs

    def forward(self, inputs):
        return self.model(**inputs, output_hidden_states=True, return_dict=True)

    def extract_embeddings(self, raw_outputs, inputs):
        last_hidden_state = raw_outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            emb = last_hidden_state.mean(dim=1)
            return emb.cpu().numpy()

        attention_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(
            last_hidden_state * attention_mask_expanded, dim=1
        )
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings.cpu().numpy()


class ItemMultimodalBatchExtractor:
    """
    Batch extractor that supports sharding by rank for multi-GPU runs.

    Usage pattern (on each process): set LOCAL_RANK / RANK / WORLD_SIZE env vars
    (torchrun will set these) and run the same script. Each process will only
    process a subset (shard) of the items.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = None,
        dataset: str = "Instruments",
        batch_size: int = 8,
        mode: str = "orig",
        include_image: bool = True,
        image_only: bool = False,
    ) -> None:
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode
        # whether to include images when constructing inputs (immutable per instance)
        self.include_image = include_image
        # whether this extractor should run in image-only mode
        self.image_only = image_only
        logger.info(
            f"Init extractor model={self.model_name} device={self.device} batch_size={self.batch_size} mode={self.mode} include_image={self.include_image} image_only={self.image_only}"
        )

        # Load model on the assigned device
        self._load_model()

    def _load_model(self) -> None:
        # choose dtype to reduce memory when on CUDA
        dtype = (
            torch.float16 if self.device.startswith("cuda") else torch.float32
        )
        logger.info(
            f"Loading model {self.model_name} with dtype={dtype} on {self.device}"
        )
        # Use ModelWrapper factory to create appropriate wrapper (Qwen2.5 or Qwen2)
        try:
            self.wrapper = ModelWrapper.from_pretrained(
                self.model_name, self.device, dtype
            )
            # keep references for backward compatibility where code expects self.processor / self.model
            self.processor = getattr(self.wrapper, "processor", None)
            self.model = getattr(self.wrapper, "model", None)
            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception:
            logger.exception(
                "Failed to load wrapper/model; falling back to direct load"
            )
            # fallback: try loading Qwen2.5 directly as before
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=dtype, device_map={"": self.device}
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model.eval()

    def load_dataset_info(
        self, dataset_path: str
    ) -> tuple[dict[str, int], dict[int, str], dict[str, Any]]:
        item2id_path = os.path.join(dataset_path, f"{self.dataset}.item2id")
        item2id: dict[str, int] = {}
        id2item: dict[int, str] = {}

        with open(item2id_path) as f:
            for line in f:
                item_id, num_id = line.strip().split("\t")
                item2id[item_id] = int(num_id)
                id2item[int(num_id)] = item_id

        item_info_path = os.path.join(
            dataset_path, f"{self.dataset}.item_enriched_v2.json"
        )
        with open(item_info_path) as f:
            item_info = json.load(f)

        return item2id, id2item, item_info

    def construct_item_text(self, item_data: dict[str, Any]) -> str:
        # Build text differently depending on extractor mode
        parts = []
        mode = getattr(self, "mode", "orig")

        # common fields
        if item_data.get("title"):
            parts.append(f"Title: {item_data['title']}")
        if item_data.get("brand"):
            parts.append(f"Brand: {item_data['brand']}")

        if mode == "orig":
            if item_data.get("categories"):
                parts.append(f"Categories: {item_data['categories']}")
            if (
                item_data.get("description")
                and item_data["description"].strip()
            ):
                parts.append(f"Description: {item_data['description']}")

        elif mode == "enhanced":
            # include enhanced components preferentially
            if item_data.get("enhanced_title"):
                parts.insert(
                    0, f"Enhanced Title: {item_data['enhanced_title']}"
                )
            if item_data.get("tags") and isinstance(item_data["tags"], list):
                parts.append(f"Tags: {', '.join(item_data['tags'])}")
            if item_data.get("highlights") and isinstance(
                item_data["highlights"], list
            ):
                parts.append(
                    f"Highlights: {', '.join(item_data['highlights'])}"
                )
            if item_data.get("characteristics") and isinstance(
                item_data["characteristics"], list
            ):
                parts.append(
                    f"Characteristics: {', '.join(item_data['characteristics'])}"
                )
            if (
                item_data.get("description")
                and item_data["description"].strip()
            ):
                parts.append(f"Description: {item_data['description']}")

        elif mode == "orig_enhanced":
            # combine both original and enhanced info
            if item_data.get("categories"):
                parts.append(f"Categories: {item_data['categories']}")
            if (
                item_data.get("description")
                and item_data["description"].strip()
            ):
                parts.append(f"Description: {item_data['description']}")
            if item_data.get("enhanced_title"):
                parts.append(f"Enhanced Title: {item_data['enhanced_title']}")
            if item_data.get("tags") and isinstance(item_data["tags"], list):
                parts.append(f"Tags: {', '.join(item_data['tags'])}")

        else:
            # fallback: include whatever is available
            if item_data.get("categories"):
                parts.append(f"Categories: {item_data['categories']}")
            if (
                item_data.get("description")
                and item_data["description"].strip()
            ):
                parts.append(f"Description: {item_data['description']}")

        return " | ".join(parts)

    def load_item_image(self, image_path: str):
        if not os.path.exists(image_path):
            return None
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    def _build_messages_for_item(self, item_data: dict[str, Any], image):
        """
        Build the `messages` payload for a single item given optional image.
        Centralizes the logic for text-only / image-only / multimodal modes.
        Returns a list suitable to be appended to batch_messages.
        """
        text = self.construct_item_text(item_data)
        image_only_mode = self.image_only

        # if we have an image available
        if image is not None:
            if image_only_mode:
                return [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": image}],
                    }
                ]
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": f"The item's text information: {text}",
                        },
                    ],
                }
            ]

        # no image available
        if image_only_mode:
            # fallback to a short placeholder text to avoid empty inputs
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "[No image available for this item]",
                        }
                    ],
                }
            ]

        # default: text-only
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The item's text information: {text}",
                    }
                ],
            }
        ]

    def prepare_batch(
        self,
        items: list[tuple[int, dict[str, Any]]],
        id2item: dict[int, str],
        image_dir: str,
    ):
        batch_messages = []
        batch_meta = []
        for num_id, item_data in items:
            item_id = id2item[
                num_id
            ]  # This line remains for context but item_id is not used
            image_path = os.path.join(image_dir, f"{item_id}.jpg")
            image = None
            if self.include_image:
                image = self.load_item_image(image_path)

            messages = self._build_messages_for_item(item_data, image)

            batch_messages.append(messages)
            batch_meta.append(
                {
                    "num_id": num_id,
                    "item_id": item_id,
                    "text": self.construct_item_text(item_data),
                    "has_image": image is not None,
                }
            )

        return batch_messages, batch_meta

    def extract_batch_embeddings(self, batch_messages: list[Any]):
        # Delegate preprocessing/forward/extraction to wrapper if available
        if hasattr(self, "wrapper") and self.wrapper is not None:
            inputs = self.wrapper.preprocess(batch_messages)
            with torch.no_grad():
                raw_outputs = self.wrapper.forward(inputs)
            emb = self.wrapper.extract_embeddings(raw_outputs, inputs)
            return emb

        # Fallback to previous behavior using self.processor / self.model
        texts = []
        images_all = []
        for messages in batch_messages:
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            images, _ = process_vision_info(messages)
            images_all.append(images if images else None)

        imgs = None if all(im is None for im in images_all) else images_all

        inputs = self.processor(
            text=texts, images=imgs, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs, output_hidden_states=True, return_dict=True
            )

        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(
            last_hidden_state * attention_mask_expanded, dim=1
        )
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings.cpu().numpy()

    def shard_items(
        self, all_items: list[tuple[int, dict[str, Any]]]
    ) -> list[tuple[int, dict[str, Any]]]:
        # single-process: return all items unchanged
        return all_items

    def run(self, dataset_path: str, output_path: str):
        # If outputs already exist, skip to avoid recomputation.
        try:
            npy_path = output_path.replace(".json", ".npy")
            if os.path.exists(output_path) and os.path.exists(npy_path):
                logger.info(f"Output already exists, skipping: {output_path}")
                # load minimal info to be compatible: return empty dict and no failures
                return {}, []
        except Exception:
            # if any issue checking, proceed normally
            pass

        # load dataset info
        item2id, id2item, item_info = self.load_dataset_info(dataset_path)
        image_dir = os.path.join(dataset_path, "images")

        # prepare all items list (sorted by numeric id)
        all_items: list[tuple[int, dict[str, Any]]] = []
        for num_id_str, item_data in item_info.items():
            num_id = int(num_id_str)
            all_items.append((num_id, item_data))
        all_items.sort(key=lambda x: x[0])

        # single-process: process all items
        my_items = all_items
        logger.info(f"Processing {len(my_items)} items on device {self.device}")

        representations: dict[int, dict[str, Any]] = {}
        failed_items: list[int] = []

        # iterate in batches
        indices = list(range(0, len(my_items), self.batch_size))
        iterable = tqdm(indices, desc="Extracting batches")

        for i in iterable:
            batch = my_items[i : i + self.batch_size]
            batch_messages, batch_meta = self.prepare_batch(
                batch, id2item, image_dir
            )
            try:
                embeddings = self.extract_batch_embeddings(batch_messages)
                # embeddings shape: (B, D)
                for j, meta in enumerate(batch_meta):
                    rep = embeddings[j]
                    # ensure 1D
                    rep = np.asarray(rep)
                    if rep.ndim == 2 and rep.shape[0] == 1:
                        rep = rep.squeeze(0)
                    representations[meta["num_id"]] = {
                        "item_id": meta["item_id"],
                        "representation": rep.tolist(),
                        "has_image": meta["has_image"],
                        "text": meta["text"],
                    }
                logger.info(f"Processed batch {i // self.batch_size + 1}")
            except Exception as e:
                logger.exception(
                    f"Batch extraction failed at batch starting index {i}: {e}"
                )
                for num_id, _ in batch:
                    failed_items.append(num_id)
            finally:
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    gc.collect()

        # save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(representations, f, indent=2)

        # Debug: save a small sample of constructed texts to help verify mode differences
        try:
            sample_path = output_path + ".texts.txt"
            with open(sample_path, "w") as sf:
                sf.write(f"mode: {getattr(self, 'mode', 'orig')}\n")
                keys_sorted = sorted(representations.keys())
                for k in keys_sorted[:5]:
                    rep = representations[k]
                    sf.write(
                        f"num_id={k} item_id={rep.get('item_id')} has_image={rep.get('has_image')}\n"
                    )
                    sf.write(rep.get("text", "") + "\n---\n")
        except Exception:
            logger.exception("Failed to write sample texts for debug")

        # save numpy in same order of numeric ids present in this shard
        keys_sorted = sorted(representations.keys())
        matrix = np.array(
            [representations[k]["representation"] for k in keys_sorted]
        )
        np.save(output_path.replace(".json", ".npy"), matrix)

        logger.info(
            f"Finished. Saved {len(representations)} items to {output_path}"
        )
        return representations, failed_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct"
    )
    parser.add_argument("--out-dir", type=str, default="reps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--mode", type=str, default="orig", help="orig, enhanced, orig_enhanced"
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="If set, do not include images in the extractor inputs (force text-only).",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="If set, run in image_only mode: no text, only image inputs (if available).",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset_path = os.path.abspath(os.path.join("data", args.dataset))
    out_dir = os.path.abspath(os.path.join(dataset_path, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Make output path include mode and image flag for clarity (so we can generate multiple combos)
    img_flag = "noimg" if args.no_image else "img"
    if args.image_only:
        mode_tag = "image_only"
    else:
        mode_tag = args.mode
    out_path = os.path.join(
        out_dir, f"{args.model.replace('/', '_')}_{mode_tag}_{img_flag}.json"
    )

    extractor = ItemMultimodalBatchExtractor(
        model_name=args.model,
        device=device,
        dataset=args.dataset,
        batch_size=args.batch_size,
        mode=args.mode,
        include_image=(not args.no_image),
        image_only=args.image_only,
    )

    extractor.run(dataset_path=dataset_path, output_path=out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
