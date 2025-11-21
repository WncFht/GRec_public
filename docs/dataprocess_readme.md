1. 其他文件基本没有动过，主要就是加载数据集，用 api enrich 文本，下载图片等。
2. 增强了 qwen_embeddings.py 但是没有增强 qwen_embedding_batch.py 所以使用还是用 qwen_embeddings.py 就行

```python
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
```

可以选择不同的 数据集，不同模型（主要就是 qwen 和 llava 的支持），不同的 batch_size 和 选择什么文本，要不要图片。最后一个 --image-only 会覆盖 --mode 和 --on-image

scipts/extract_rep.py 会生成上述 7 种情况的所有 emb