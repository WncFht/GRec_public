import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from res_kmeans import ResKmeans


def _load_ids(ids_path: str) -> list:
    if ids_path.endswith(".json"):
        with open(ids_path, encoding="utf-8") as f:
            ids = json.load(f)
        if not isinstance(ids, list):
            raise ValueError(f"Expected a JSON list in {ids_path}")
        return ids

    if ids_path.endswith(".npy"):
        ids = np.load(ids_path, allow_pickle=True)
        ids = np.squeeze(ids)
        return ids.tolist()

    if ids_path.endswith(".txt"):
        with open(ids_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    raise ValueError(f"Unsupported ids_path format: {ids_path}")


def _load_embeddings_npy(emb_path: str) -> np.ndarray:
    arr = np.load(emb_path, mmap_mode="r")
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D array after squeeze, got shape={arr.shape} from {emb_path}"
        )
    return arr


def _load_embeddings_parquet(emb_path: str) -> tuple[list, np.ndarray]:
    import pandas as pd

    df = pd.read_parquet(emb_path)
    if "embedding" not in df.columns:
        raise ValueError("Parquet input must contain an 'embedding' column")
    pids = df["pid"].tolist() if "pid" in df.columns else list(range(len(df)))
    emb = np.stack(df["embedding"].values)
    return pids, emb


def load_model(model_path: str) -> ResKmeans:
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, ResKmeans):
        model = checkpoint
        return model

    if not isinstance(checkpoint, dict):
        raise ValueError("Unknown checkpoint format")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    n_layers = sum(1 for k in state_dict.keys() if k.startswith("centroids."))
    first_centroid = state_dict["centroids.0"]
    codebook_size, dim = first_centroid.shape

    model = ResKmeans(n_layers=n_layers, codebook_size=codebook_size, dim=dim)
    model.load_state_dict(state_dict)
    return model


def codes_to_tokens(codes: np.ndarray) -> list[list[str]]:
    # codes: [N, L]
    n_layers = codes.shape[1]
    if n_layers > 26:
        raise ValueError(f"Too many layers for a-z prefixes: n_layers={n_layers}")
    out: list[list[str]] = []
    for row in codes:
        toks = []
        for layer_idx, c in enumerate(row.tolist()):
            prefix = chr(ord("a") + layer_idx)
            toks.append(f"<{prefix}_{int(c)}>")
        out.append(toks)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Export OpenOneRec-style index JSON from embeddings."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="ResKmeans checkpoint path",
    )
    parser.add_argument(
        "--emb_path",
        type=str,
        required=True,
        help="Embedding path (.npy or .parquet)",
    )
    parser.add_argument(
        "--ids_path",
        type=str,
        default=None,
        help="Optional item-id list path (.ids.json from amazon_text2emb.py is recommended).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSON path (e.g., /path/Dataset.index_xxx.json)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10000, help="Inference batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=None,
        help="Number of layers to use (default: all layers).",
    )
    args = parser.parse_args()

    model = load_model(args.model_path).to(args.device)
    model.eval()
    use_layers = args.n_layers or model.n_layers

    emb_path = args.emb_path
    ids = None

    if args.ids_path is not None:
        ids = _load_ids(args.ids_path)
    else:
        # Auto-pick ids saved by amazon_text2emb.py if present
        if emb_path.endswith(".npy"):
            guess = emb_path[:-4] + ".ids.json"
        else:
            guess = emb_path + ".ids.json"
        if os.path.exists(guess):
            ids = _load_ids(guess)

    if emb_path.endswith(".parquet"):
        pids, emb = _load_embeddings_parquet(emb_path)
        if ids is None:
            ids = pids
    else:
        emb = _load_embeddings_npy(emb_path)

    n = int(emb.shape[0])
    if ids is None:
        ids = list(range(n))
    if len(ids) != n:
        raise ValueError(
            f"ids length mismatch: len(ids)={len(ids)} vs num_embeddings={n}"
        )

    # Clean NaN/Inf once, and keep ids in sync
    finite_mask = np.isfinite(emb).all(axis=1)
    num_bad = int(n - finite_mask.sum())
    if num_bad > 0:
        print(
            f"[warn] Found {num_bad} rows with NaN/Inf in embeddings, dropping them for export."
        )
        bad_indices = np.nonzero(~finite_mask)[0][:10]
        print(f"[warn] First bad row indices (up to 10): {bad_indices.tolist()}")
        emb = emb[finite_mask]
        # Keep ids aligned with embeddings
        ids = [ids[i] for i in range(len(ids)) if finite_mask[i]]
        n = int(emb.shape[0])
        print(f"[info] Clean embeddings shape for export: {emb.shape}, ids: {len(ids)}")

    out: dict[str, list[str]] = {}
    seen_codes = set()
    dup = 0

    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch_np = np.asarray(emb[start:end], dtype=np.float32)
            batch = torch.from_numpy(batch_np).to(args.device)
            codes = model.encode(batch, n_layers=use_layers).cpu().numpy()
            toks = codes_to_tokens(codes)

            for item_id, token_list in zip(ids[start:end], toks, strict=False):
                key = str(item_id)
                out[key] = token_list
                code_key = tuple(token_list)
                if code_key in seen_codes:
                    dup += 1
                else:
                    seen_codes.add(code_key)

    os.makedirs(str(Path(args.output_path).parent), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    collision_rate = dup / max(1, len(out))
    print(f"Saved index JSON: {args.output_path}")
    print(
        f"Items: {len(out):,}, n_layers: {use_layers}, collision_rate: {collision_rate:.6f}"
    )


if __name__ == "__main__":
    main()
