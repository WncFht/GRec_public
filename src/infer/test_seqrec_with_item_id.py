import os

from src.config import parse_args
from src.data import SeqRecDataset

args = parse_args()
dataset = SeqRecDataset(args, mode="test")
print(dataset[0])

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from src.collator import UnifiedTestCollator

ckpt_path = os.environ.get("CKPT_PATH")
model_type = os.environ.get("MODEL_TYPE")
tokenizer = AutoProcessor.from_pretrained(ckpt_path)
collator = UnifiedTestCollator(args, processor_or_tokenizer=tokenizer)
if model_type == "qwen2_5_vl":
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_path, trust_remote_code=True
    )
elif model_type == "qwen_2_vl":
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ckpt_path, trust_remote_code=True
    )
model.eval()
model.to("cuda")

length = len(dataset)
for i in range(length - 5, length):
    batch = collator([dataset[i]])
    inputs = batch[0]
    print("Inputs:", "=" * 50)
    print(batch[1])
    # inputs: set(input, target, item_ids)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # use beam search
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=4,
        # max_length=10,
        # prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=10,
        num_return_sequences=10,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True,
    )
    output_ids = output["sequences"]
    scores = output["sequences_scores"]

    print("Outputs:", "=" * 50)
    # decode all the results
    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # extract the output behind the "Response:"
    for i, o in enumerate(output_texts):
        o = o.split("Response:")[-1]
        print(o, "|", batch[1], "|", float(scores[i]))
