#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np


def _load_npy_mmap(path: str) -> np.ndarray:
    """Load a .npy file with mmap and ensure 2D array."""
    arr = np.load(path, mmap_mode="r")
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D array after squeeze, got shape={arr.shape} from {path}"
        )
    return arr


def _expand_paths(patterns):
    """Expand globs like /xx/yy/*.npy into concrete file list."""
    out = []
    for p in patterns:
        # 有通配符
        if any(ch in p for ch in ["*", "?", "["]):
            pp = Path(p)
            out.extend(str(x) for x in sorted(pp.parent.glob(pp.name)))
        else:
            out.append(p)
    return out


def check_file(path: str, emb_dim: int | None = None) -> dict:
    """Check single .npy file for NaN/Inf; return stats."""
    arr = _load_npy_mmap(path)
    if emb_dim is not None:
        arr = arr[:, :emb_dim]

    print(f"\nChecking {path}")
    print(f"  shape={arr.shape}, dtype={arr.dtype}")

    # 基本统计
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    nan_count = int(nan_mask.sum())
    inf_count = int(inf_mask.sum())

    # 哪些行有任何 NaN/Inf
    finite_mask = np.isfinite(arr)
    good_row_mask = finite_mask.all(axis=1)
    bad_row_mask = ~good_row_mask
    num_bad_rows = int(bad_row_mask.sum())

    if nan_count == 0 and inf_count == 0:
        print("  OK: no NaN/Inf found.")
    else:
        print(f"  NaN count: {nan_count}")
        print(f"  Inf count: {inf_count}")
        print(f"  Rows with any NaN/Inf: {num_bad_rows}")

        # 打印前几个坏行的 index，方便你回溯源数据
        bad_indices = np.nonzero(bad_row_mask)[0][:10]
        print(f"  First bad row indices (up to 10): {bad_indices.tolist()}")

    return {
        "path": path,
        "shape": arr.shape,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "bad_rows": num_bad_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check .npy embedding files for NaN/Inf."
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of .npy paths or globs (e.g. /path/Arts/*.npy).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Optional embedding dim; if set, only first dim columns are checked.",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.data_paths)
    if not paths:
        raise SystemExit("No files found for given --data_paths.")

    print(f"Total files to check: {len(paths)}")

    total_nan = 0
    total_inf = 0
    total_bad_rows = 0

    for p in paths:
        if not p.endswith(".npy"):
            print(f"\nSkip non-npy file: {p}")
            continue
        stats = check_file(p, emb_dim=args.dim)
        total_nan += stats["nan_count"]
        total_inf += stats["inf_count"]
        total_bad_rows += stats["bad_rows"]

    print("\n========== Summary ==========")
    print(f"Total NaN: {total_nan}")
    print(f"Total Inf: {total_inf}")
    print(f"Total rows with any NaN/Inf: {total_bad_rows}")


if __name__ == "__main__":
    main()
