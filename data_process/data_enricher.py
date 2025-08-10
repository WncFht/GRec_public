import argparse
import base64
import json
import logging
import os
import time
from typing import Any

from openai import OpenAI
from tqdm import tqdm


class TextEnricher:
    def __init__(self):
        self.client = OpenAI(
            api_key="30a1f4e6015e45fe8978265697a77150",
            base_url="https://runway.devops.xiaohongshu.com/openai",
            default_headers={"api-key": "30a1f4e6015e45fe8978265697a77150"},
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_item_prompt(self) -> str:
        """Get standardized prompt template for item enrichment"""
        return """You are an expert product content creator for e-commerce recommendation systems. 
    Based on the image and original product information, please generate:
    
    1. An optimized, more engaging title (maintaining the original brand and product type)
    2. 5-10 relevant tags that capture key attributes
    3. 3-5 product highlights that emphasize unique selling points
    4. 3-5 product characteristics with technical details
    
    CRITICAL CONSTRAINTS - YOU MUST FOLLOW THESE RULES:
    - Only use information that is explicitly visible in the image and stated in the original product information
    - Do NOT invent, assume, or add any features, specifications, or characteristics not present in the source materials
    - All generated content must be factually verifiable from the provided sources
    - When in doubt about any detail, err on the side of being conservative and accurate

    Original product information:
    Title: {title}
    Brand: {brand}
    Categories: {categories}
    Description: {description}
    
    Please respond in strict JSON format with these fields:
    {{
        "optimized_title": "optimized title here",
        "tags": ["tag1", "tag2", ...],
        "highlights": ["highlight1", "highlight2", ...],
        "characteristics": ["characteristic1", "characteristic2", ...]
    }}"""

    def create_multimodal_prompt(
        self, item_info: dict[str, Any], image_path: str
    ) -> list[dict[str, Any]]:
        """Create a multimodal prompt with image and text"""
        base64_image = self.encode_image(image_path)

        prompt = self.get_item_prompt()
        prompt = prompt.format(
            title=item_info.get("title", ""),
            brand=item_info.get("brand", ""),
            categories=item_info.get("categories", ""),
            description=item_info.get("description", ""),
        )

        return [
            {
                "role": "system",
                "content": "You are an expert product content creator for e-commerce recommendation systems.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

    def create_text_only_prompt(
        self, item_info: dict[str, Any]
    ) -> list[dict[str, str]]:
        # Extract item information
        title = item_info.get("title", "")
        brand = item_info.get("brand", "")
        categories = item_info.get("categories", "")
        description = item_info.get("description", "")

        prompt_text = f"""You are an expert product content creator for e-commerce recommendation systems. 
        Based on the original product information provided, please generate:
        
        1. An optimized, more engaging title (maintaining the original brand and product type)
        2. 5-10 relevant tags that capture key attributes
        3. 3-5 product highlights that emphasize unique selling points
        4. 3-5 product characteristics with technical details
        
        CRITICAL CONSTRAINTS - YOU MUST FOLLOW THESE RULES:
        - Only use information that is stated in the original product information
        - Do NOT invent, assume, or add any features, specifications, or characteristics not present in the source materials
        - All generated content must be factually verifiable from the provided sources
        - When in doubt about any detail, err on the side of being conservative and accurate

        Original product information:
        Title: {title}
        Brand: {brand}
        Categories: {categories}
        Description: {description}
        
        Please respond in strict JSON format with these fields:
        {{
            "optimized_title": "optimized title here",
            "tags": ["tag1", "tag2", ...],
            "highlights": ["highlight1", "highlight2", ...],
            "characteristics": ["characteristic1", "characteristic2", ...]
        }}"""

        return [{"role": "user", "content": prompt_text}]

    def enrich_item_text(
        self, item_info: dict[str, Any], image_path: str, num_id: int
    ) -> dict[str, Any]:
        """Generate enriched text information using GPT-4"""
        try:
            # Check if image exists
            has_image = os.path.exists(image_path)

            if has_image:
                # Create multimodal prompt with image
                messages = self.create_multimodal_prompt(item_info, image_path)
            else:
                # Create text-only prompt
                messages = self.create_text_only_prompt(item_info)

            # Call GPT-4 API
            completion = self.client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=500,
                extra_query={"api-version": "2024-12-01-preview"},
            )

            # Parse response
            enriched_data = json.loads(completion.choices[0].message.content)

            # Add enriched data to original item info
            item_info["enhanced_title"] = enriched_data.get(
                "optimized_title", item_info["title"]
            )
            item_info["tags"] = enriched_data.get("tags", [])
            item_info["highlights"] = enriched_data.get("highlights", [])
            item_info["characteristics"] = enriched_data.get(
                "characteristics", []
            )
            item_info["has_image"] = has_image  # Track whether image was used

            return item_info

        except Exception as e:
            logging.exception(f"Failed to enrich item {num_id}: {e}")
            return item_info  # Return original if enrichment fails

    def load_dataset_info(
        self, dataset_path: str, dataset_name: str
    ) -> dict[str, dict]:
        """Load original item information from JSON file"""
        item_info_path = os.path.join(dataset_path, f"{dataset_name}.item.json")
        with open(item_info_path) as f:
            return json.load(f)

    def load_item_ids(
        self, dataset_path: str, dataset_name: str
    ) -> dict[int, str]:
        """Load item ID mapping"""
        item2id_path = os.path.join(dataset_path, f"{dataset_name}.item2id")
        item2id = {}

        with open(item2id_path) as f:
            for line in f:
                item_id, num_id = line.strip().split("\t")
                item2id[int(num_id)] = item_id

        return item2id

    def process_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        output_path: str,
        limit: int = None,
    ):
        """Process all items in the dataset to enrich text information with incremental saving"""
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        items = self.load_dataset_info(dataset_path, dataset_name)
        item_ids = self.load_item_ids(dataset_path, dataset_name)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize or load existing enriched items
        enriched_items = {}
        if os.path.exists(output_path):
            print(f"Loading existing results from {output_path}")
            with open(output_path, encoding="utf-8") as f:
                enriched_items = json.load(f)

        failed_items = []
        failed_path = output_path.replace(".json", "_failed.json")
        if os.path.exists(failed_path):
            with open(failed_path, encoding="utf-8") as f:
                failed_items = json.load(f)

        # Process items
        image_dir = os.path.join(dataset_path, "images")
        items_to_process = (
            list(items.items())
            if limit is None
            else list(items.items())[:limit]
        )

        # Filter out already processed items
        items_to_process = [
            (num_id, item_data)
            for num_id, item_data in items_to_process
            if num_id not in enriched_items
        ]

        print(
            f"Processing {len(items_to_process)} new items for text enrichment..."
        )

        for num_id, item_data in tqdm(items_to_process, desc="Enriching items"):
            item_id = item_ids.get(int(num_id), num_id)
            image_path = os.path.join(image_dir, f"{item_id}.jpg")

            try:
                enriched_item = self.enrich_item_text(
                    item_data, image_path, num_id
                )
                enriched_items[num_id] = enriched_item

                # Incremental save after each successful processing
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(enriched_items, f, indent=2, ensure_ascii=False)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logging.exception(f"Failed to process item {num_id}: {e}")
                failed_items.append(num_id)
                enriched_items[num_id] = (
                    item_data  # Save original if enrichment fails
                )

                # Save failed items incrementally
                with open(failed_path, "w", encoding="utf-8") as f:
                    json.dump(failed_items, f, indent=2, ensure_ascii=False)

        print(
            f"Successfully enriched {len(enriched_items) - len(failed_items)} items"
        )
        print("Text enrichment completed!")
        return enriched_items, failed_items


def main(args):
    # Initialize enricher
    enricher = TextEnricher()

    # Process dataset
    dataset_path = os.path.abspath(os.path.join("data", args.dataset))
    output_path = os.path.abspath(
        os.path.join(
            "data", args.dataset, f"{args.dataset}.item_enriched_v2.json"
        )
    )

    print(f"Starting text enrichment for dataset: {args.dataset}")
    enriched_items, failed_items = enricher.process_dataset(
        dataset_path=dataset_path,
        dataset_name=args.dataset,
        output_path=output_path,
        limit=args.limit,
    )

    print(
        f"Text enrichment completed. {len(failed_items)} items failed processing."
    )
    print(f"Enhanced data saved to {output_path}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Enrich Product Text Information"
    )
    parser.add_argument(
        "--dataset", type=str, default="Instruments", help="Dataset Name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process (for testing)",
    )
    args = parser.parse_args()

    main(args)
