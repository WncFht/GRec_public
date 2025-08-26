import os

from src.config import parse_args
from src.data import SeqRectWithoutItemIDDataset_1

args = parse_args()
dataset = SeqRectWithoutItemIDDataset_1(args, mode="test")
print(dataset[0])

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.collator import UnifiedTestCollator

ckpt_path = os.environ.get("CKPT_PATH")
processor = AutoProcessor.from_pretrained(ckpt_path)
collator = UnifiedTestCollator(args, processor_or_tokenizer=processor)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ckpt_path, trust_remote_code=True
)
model.eval()
model.to("cuda")

length = len(dataset)
for i in range(length - 11, length):
    inputs = collator([dataset[i]])
    print("Inputs:", "=" * 50)
    print(dataset[i].label_text)
    inputs = {k: v.to("cuda") for k, v in inputs[0].items()}
    results = model.generate(**inputs)

    print("Outputs:", "=" * 50)
    print(processor.tokenizer.decode(results[0], skip_special_tokens=True))
