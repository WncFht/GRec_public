import argparse
import os
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from res_kmeans import ResKmeans
from tqdm import tqdm


def _load_npy_mmap(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D array after squeeze, got shape={arr.shape} from {path}"
        )
    return arr


def _allocate_sample_counts(sizes: list[int], k: int) -> list[int]:
    if k <= 0:
        return sizes

    total = sum(sizes)
    if k >= total:
        return sizes

    raw = [k * (s / total) for s in sizes]
    base = [int(np.floor(x)) for x in raw]

    # Ensure every non-empty source contributes at least 1 point (when possible)
    for i, s in enumerate(sizes):
        if s > 0 and base[i] == 0:
            base[i] = 1

    # Cap by available size
    base = [min(b, s) for b, s in zip(base, sizes, strict=False)]

    curr = sum(base)
    if curr > k:
        # Reduce from the largest allocations first (keep at least 1 if source non-empty)
        order = sorted(range(len(sizes)), key=lambda i: base[i], reverse=True)
        for i in order:
            while curr > k and base[i] > (1 if sizes[i] > 0 else 0):
                base[i] -= 1
                curr -= 1
            if curr == k:
                break
    elif curr < k:
        # Add to sources with remaining capacity, based on largest remaining capacity
        remaining = [sizes[i] - base[i] for i in range(len(sizes))]
        order = sorted(range(len(sizes)), key=lambda i: remaining[i], reverse=True)
        for i in order:
            while curr < k and remaining[i] > 0:
                base[i] += 1
                remaining[i] -= 1
                curr += 1
            if curr == k:
                break

    return base


def _read_train_data_parquet(path: str, emb_dim: int) -> np.ndarray:
    """Read training data from local parquet files/directories (embedding column)."""
    dataset = pq.ParquetDataset(path)

    fragments = list(dataset.fragments)
    random.shuffle(fragments)
    print(f"Total files: {len(fragments)}")

    embeddings = []
    current_size = 0

    for fragment in tqdm(fragments, desc="Reading files"):
        table = fragment.to_table(columns=["embedding"])
        if table.num_rows == 0:
            continue

        emb_chunk = table["embedding"].to_numpy(zero_copy_only=False)
        if emb_chunk.dtype == "object":
            emb_chunk = np.vstack(emb_chunk)

        emb_chunk = emb_chunk[:, :emb_dim].astype(np.float32)

        # Clean NaN/Inf per fragment to avoid Faiss errors
        finite_mask = np.isfinite(emb_chunk).all(axis=1)
        num_bad = int(emb_chunk.shape[0] - finite_mask.sum())
        if num_bad > 0:
            print(f"[warn] Dropping {num_bad} rows with NaN/Inf from parquet fragment")
            emb_chunk = emb_chunk[finite_mask]

        embeddings.append(emb_chunk)
        current_size += len(emb_chunk)

    if not embeddings:
        raise ValueError(f"No embeddings loaded from parquet path: {path}")

    result = np.concatenate(embeddings, axis=0)
    print(f"Final shape: {result.shape}")
    return result


def _read_train_data_npy(
    paths: list[str], emb_dim: int, max_train_points: int, seed: int
) -> np.ndarray:
    sizes = []
    mmaps = []
    for p in paths:
        arr = _load_npy_mmap(p)
        mmaps.append((p, arr))
        sizes.append(int(arr.shape[0]))

    alloc = _allocate_sample_counts(sizes, max_train_points)
    rng = np.random.default_rng(seed)

    sampled = []
    for (p, arr), take_n in zip(mmaps, alloc, strict=False):
        if take_n <= 0:
            continue
        n = arr.shape[0]
        if take_n >= n:
            chunk = np.asarray(arr[:, :emb_dim], dtype=np.float32)
        else:
            idx = rng.choice(n, size=take_n, replace=False)
            idx.sort()
            chunk = np.asarray(arr[idx, :emb_dim], dtype=np.float32)

        # Clean NaN/Inf per file
        finite_mask = np.isfinite(chunk).all(axis=1)
        num_bad = int(chunk.shape[0] - finite_mask.sum())
        if num_bad > 0:
            print(f"[warn] Dropping {num_bad} rows with NaN/Inf from {p}")
            bad_indices = np.nonzero(~finite_mask)[0][:10]
            print(
                f"[warn] First bad row indices in this file (up to 10): {bad_indices.tolist()}"
            )
            chunk = chunk[finite_mask]

        sampled.append(chunk)
        print(f"Loaded {len(chunk):,} / {n:,} from {p}")

    if not sampled:
        raise ValueError("No embeddings loaded; please check input paths.")

    result = np.concatenate(sampled, axis=0)
    print(f"Final shape: {result.shape}")
    return result


def read_train_data(
    paths: list[str], emb_dim: int, max_train_points: int, seed: int
) -> np.ndarray:
    """Load training embeddings from .npy or parquet paths (supports multiple inputs)."""
    if not paths:
        raise ValueError("paths must be a non-empty list")

    # Expand globs for convenience
    expanded: list[str] = []
    for p in paths:
        pp = Path(p)
        if any(ch in p for ch in ["*", "?", "["]):
            expanded.extend([str(x) for x in sorted(pp.parent.glob(pp.name))])
        else:
            expanded.append(p)

    npy_paths = [p for p in expanded if p.endswith(".npy")]
    other_paths = [p for p in expanded if not p.endswith(".npy")]

    if other_paths and npy_paths:
        raise ValueError("Please do not mix .npy and parquet inputs in one run.")

    if npy_paths:
        return _read_train_data_npy(npy_paths, emb_dim, max_train_points, seed)

    if len(other_paths) > 1:
        raise ValueError(
            "Parquet mode currently expects a single dataset path (file or directory)."
        )
    return _read_train_data_parquet(other_paths[0], emb_dim)


def main():
    parser = argparse.ArgumentParser(description="Train ResKmeans")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Training data path. Supports a parquet file/dir OR a single .npy file.",
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of .npy paths (or globs) to train on the union of multiple datasets.",
    )
    parser.add_argument("--model_path", type=str, required=True, help="model save path")
    parser.add_argument("--n_layers", type=int, default=3, help="number of layers")
    parser.add_argument("--codebook_size", type=int, default=8192, help="codebook size")
    parser.add_argument("--dim", type=int, default=4096, help="embedding dimension")
    parser.add_argument("--niter", type=int, default=20, help="kmeans iterations")
    parser.add_argument(
        "--max_train_points",
        type=int,
        default=0,
        help="Max total training points (only for .npy inputs). 0 means use all points.",
    )
    parser.add_argument(
        "--max_points_per_centroid",
        type=int,
        default=256,
        help="Faiss KMeans max_points_per_centroid (controls internal subsampling).",
    )
    parser.add_argument(
        "--faiss_gpu",
        action="store_true",
        help="Use Faiss GPU KMeans if available (requires faiss-gpu).",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.faiss_gpu:
        try:
            import faiss

            n_gpu = int(getattr(faiss, "get_num_gpus", lambda: 0)())
            print(f"Faiss GPUs visible: {n_gpu}")
            if n_gpu <= 0:
                print(
                    "[warn] --faiss_gpu set but faiss reports 0 GPUs; using CPU instead."
                )
                args.faiss_gpu = False
        except Exception as e:
            print(f"[warn] Failed to query faiss GPUs ({e}); using CPU instead.")
            args.faiss_gpu = False

    # Load data
    if args.data_path is not None and args.data_paths is not None:
        raise ValueError("Please use either --data_path or --data_paths, not both.")
    paths = args.data_paths if args.data_paths is not None else [args.data_path]
    if not paths or paths == [None]:
        raise ValueError("Please provide --data_path or --data_paths.")

    embeddings = read_train_data(paths, args.dim, args.max_train_points, args.seed)
    print(f"[info] Raw embeddings shape: {embeddings.shape}")

    # Safety check: ensure no NaN/Inf remain (should be very rare after per-file cleaning)
    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Final training embeddings still contain NaN/Inf after cleaning."
        )

    dim = int(embeddings.shape[1])
    if dim != args.dim:
        print(
            f"[warn] --dim={args.dim} but loaded embeddings dim={dim}; using dim={dim} for training."
        )
        args.dim = dim

    # Create and train model
    model = ResKmeans(
        n_layers=args.n_layers,
        codebook_size=args.codebook_size,
        dim=dim,
        extra_kmeans_config={
            "niter": args.niter,
            "seed": args.seed,
            "verbose": True,
            "gpu": args.faiss_gpu,
            "max_points_per_centroid": args.max_points_per_centroid,
        },
    )
    model.train_kmeans(embeddings)

    # Save model
    os.makedirs(args.model_path, exist_ok=True)
    save_path = os.path.join(args.model_path, "model.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "n_layers": args.n_layers,
                "codebook_size": args.codebook_size,
                "dim": dim,
                "data_paths": paths,
                "max_train_points": args.max_train_points,
                "niter": args.niter,
                "seed": args.seed,
                "max_points_per_centroid": args.max_points_per_centroid,
                "faiss_gpu": args.faiss_gpu,
            },
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
