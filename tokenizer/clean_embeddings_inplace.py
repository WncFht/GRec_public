#!/usr/bin/env python

import argparse
import json
import os

import numpy as np


def clean_one_pair(emb_path: str, ids_path: str | None = None, dry_run: bool = False):
    emb_path = os.path.abspath(emb_path)
    print(f"\n=== Checking {emb_path} ===")

    arr = np.load(emb_path, mmap_mode="r")
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    n, d = arr.shape
    print(f"  shape={arr.shape}, dtype={arr.dtype}")

    finite_mask = np.isfinite(arr).all(axis=1)
    num_bad = int(n - finite_mask.sum())
    if num_bad == 0:
        print("  OK: no NaN/Inf rows found.")
        return

    bad_idx = np.nonzero(~finite_mask)[0]
    print(f"  Found {num_bad} rows with NaN/Inf.")
    print(f"  Bad row indices (up to 20): {bad_idx[:20].tolist()}")

    if dry_run:
        print("  [dry-run] Not modifying files.")
        return

    # 1) 写回干净的 embeddings
    clean_arr = np.asarray(arr[finite_mask], dtype=np.float32)
    backup_emb = emb_path + ".bak"
    print(f"  Backing up original embeddings to {backup_emb}")
    os.rename(emb_path, backup_emb)
    np.save(emb_path, clean_arr)
    print(f"  Saved cleaned embeddings to {emb_path} (shape={clean_arr.shape})")

    # 2) 如果有 ids，则同步删除对应行
    if ids_path is not None:
        ids_path = os.path.abspath(ids_path)
        print(f"  Syncing ids with mask: {ids_path}")
        if ids_path.endswith(".json"):
            with open(ids_path, encoding="utf-8") as f:
                ids = json.load(f)
        elif ids_path.endswith(".npy"):
            ids = np.load(ids_path, allow_pickle=True).tolist()
        else:
            raise ValueError(f"Unsupported ids format: {ids_path}")

        if len(ids) != n:
            print(
                f"  [warn] len(ids)={len(ids)} != num_rows={n}, "
                "skip ids trimming to avoid misalignment."
            )
            return

        if isinstance(ids, np.ndarray):
            ids = ids.tolist()

        clean_ids = [id_ for id_, keep in zip(ids, finite_mask, strict=False) if keep]

        backup_ids = ids_path + ".bak"
        print(f"  Backing up original ids to {backup_ids}")
        os.rename(ids_path, backup_ids)

        if ids_path.endswith(".json"):
            with open(ids_path, "w", encoding="utf-8") as f:
                json.dump(clean_ids, f, ensure_ascii=False)
        elif ids_path.endswith(".npy"):
            np.save(ids_path, np.array(clean_ids, dtype=object))

        print(f"  Saved cleaned ids to {ids_path} (len={len(clean_ids)})")


def main():
    parser = argparse.ArgumentParser(
        description="Clean NaN/Inf rows from embedding .npy files (and optional ids)."
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of spec: 'emb_path[:ids_path]'. "
            "Example: /path/Tools.emb.npy:/path/Tools.emb.ids.json"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report NaN/Inf rows, do not modify files.",
    )
    args = parser.parse_args()

    for spec in args.pairs:
        if ":" in spec:
            emb_path, ids_path = spec.split(":", 1)
            ids_path = ids_path or None
        else:
            emb_path, ids_path = spec, None

        if not os.path.exists(emb_path):
            print(f"[skip] emb_path not exists: {emb_path}")
            continue
        if ids_path is not None and not os.path.exists(ids_path):
            print(f"[warn] ids_path not exists, will ignore: {ids_path}")
            ids_path = None

        clean_one_pair(emb_path, ids_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
