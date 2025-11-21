# benchmark_mmseqrec_dataloader.py
import argparse
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.data import DataLoader
from tqdm import tqdm  # << 新增
from transformers import AutoProcessor

from src.collator import MultiModalCollator
from src.data import MultimodalSeqRecDataset
from src.parser import parse_dataset_args, parse_global_args

# ------------------ 公共参数 ------------------
parser = argparse.ArgumentParser()
parser = parse_dataset_args(parser)
parser = parse_global_args(parser)
args = parser.parse_args()

BATCH_SIZE = 2
EPOCHS = 1
NUM_WORKERS_LIST = [2, 4, 6, 8, 10, 12]

# ------------------ 构造 Dataset 与 Collator ------------------
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
train_ds = MultimodalSeqRecDataset(args, mode="train")
collator = MultiModalCollator(args, processor_or_tokenizer=processor)
print(len(train_ds), "samples in dataset")


def run_once(num_workers: int):
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    start = time.time()
    for epoch in range(EPOCHS):
        # 用 tqdm 包裹 enumerate，实时显示进度
        for _ in tqdm(
            loader, desc=f"Epoch {epoch + 1}/{EPOCHS} (nw={num_workers})"
        ):
            pass  # 只做加载
    elapsed = time.time() - start
    print(f"num_workers={num_workers:2d}  ->  {elapsed:.2f} s")
    return elapsed


# ------------------ 主循环 ------------------
if __name__ == "__main__":
    print("Benchmarking MultimodalSeqRecDataset + MultiModalCollator ...")
    for nw in NUM_WORKERS_LIST:
        run_once(nw)
