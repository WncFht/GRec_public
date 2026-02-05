import argparse

from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from src.collator import UnifiedTestCollator
from src.data import TextEnrichDataset
from src.parser import (
    parse_dataset_args,
    parse_global_args,
    parse_test_args,
)


def main(args: argparse.Namespace):
    dataset = TextEnrichDataset(args, mode="test")
    print(dataset[0])

    ckpt_path = args.ckpt_path
    model_type = args.model_type
    tokenizer = AutoProcessor.from_pretrained(ckpt_path, use_fast=True)
    collator = UnifiedTestCollator(args, processor_or_tokenizer=tokenizer)
    if model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt_path, trust_remote_code=True
        )
    elif model_type == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            ckpt_path, trust_remote_code=True
        )
    elif model_type == "llava_onevision":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            ckpt_path, trust_remote_code=True
        )
    model.eval()
    model.to("cuda")

    length = len(dataset)
    for i in range(length - 5, length):
        batch = collator([dataset[i]])
        inputs = batch[0]
        # inputs: set(input, target, item_ids)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # use beam search
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=2048,
            # do_sample=True,
            # temperature=0.001,
            # top_p=0.9
        )

        # decode all the results
        output_texts = tokenizer.batch_decode(output, skip_special_tokens=False)

        # extract the output behind the "Response:"
        for i, o in enumerate(output_texts):
            o = o.split("assistant")[-1]
            print("=" * 10, "Ours:", "=" * 10)
            print(o)
            print("=" * 10, "Ground Truth:", "=" * 10)
            print(batch[1][0])
            print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    main(args)
