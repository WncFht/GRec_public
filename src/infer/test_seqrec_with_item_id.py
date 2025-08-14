import os

from src.config import parse_args
from src.data import SeqRecDataset

args = parse_args()
dataset = SeqRecDataset(args, mode="test")
print(dataset[0])

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.collator import UnifiedTestCollator

ckpt_path = os.environ.get("CKPT_PATH")
tokenizer = AutoProcessor.from_pretrained(ckpt_path)
collator = UnifiedTestCollator(args, processor_or_tokenizer=tokenizer)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ckpt_path, trust_remote_code=True
)
model.eval()
model.to("cuda")

length = len(dataset)
for i in range(length - 5, length):
    inputs = collator([dataset[i]])
    print("Inputs:", "=" * 50)
    print(dataset[i].label_text)
    # inputs: set(input, target, item_ids)
    inputs = {k: v.to("cuda") for k, v in inputs[0].items()}

    # use beam search
    outputs = model.generate(
        **inputs,
        num_beams=10,
        # max_new_tokens=10,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True,
    )
    output_ids = outputs["sequences"]
    scores = outputs["sequences_scores"]

    print("Outputs:", "=" * 50)
    # decode all the results
    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for i, o in enumerate(output_texts):
        print(o, "|", dataset[i].label_text, "|", float(scores[i]))
