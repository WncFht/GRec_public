import os

from src.config import parse_args
from src.data import SeqRecDataset

args = parse_args()
dataset = SeqRecDataset(args, mode="test")
print(dataset[0])

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.collator import MultiModalCollator

ckpt_path = os.environ.get("CKPT_PATH")
tokenizer = AutoProcessor.from_pretrained(ckpt_path)
collator = MultiModalCollator(args, processor_or_tokenizer=tokenizer)
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
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    results = model.generate(**inputs)

    print("Outputs:", "=" * 50)
    print(tokenizer.decode(results[0], skip_special_tokens=True))
