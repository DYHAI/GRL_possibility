#!/usr/bin/env python3
"""Pack member NPZ files into a single PyTorch .pt dataset for fast U-Net training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.kelvin_helmholtz.kh_dataset import (
    VAR_NAMES,
    compute_norm_stats,
    discover_members,
    load_members,
    split_members_three_way,
)


def build_dataset_pt(
    data_dir: Path,
    out_path: Path,
    *,
    max_members: int = 200,
    n_val: int = 10,
    n_test: int = 20,
    split_seed: int = 42,
    dtype: str = "float16",
) -> dict:
    member_ids = discover_members(data_dir)
    if max_members > 0:
        member_ids = member_ids[:max_members]
    if len(member_ids) < n_val + n_test + 1:
        raise SystemExit(
            f"Need >= {n_val + n_test + 1} members, found {len(member_ids)}"
        )

    train_ids, val_ids, test_ids = split_members_three_way(
        member_ids, n_val=n_val, n_test=n_test, seed=split_seed
    )
    all_members = load_members(data_dir, member_ids)
    train_members = {mid: all_members[mid] for mid in train_ids}
    mean, std = compute_norm_stats(train_members)

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    chunks: list[np.ndarray] = []
    times_chunks: list[np.ndarray] = []
    member_slices: dict[int, tuple[int, int]] = {}
    offset = 0

    for mid in member_ids:
        X = all_members[mid].X
        times = all_members[mid].times
        n = len(X)
        chunks.append(X)
        times_chunks.append(times)
        member_slices[mid] = (offset, offset + n)
        offset += n

    X_all = np.concatenate(chunks, axis=0)
    times_all = np.concatenate(times_chunks, axis=0)

    payload = {
        "version": 1,
        "dtype": dtype,
        "data_dir": str(data_dir),
        "split_seed": split_seed,
        "member_ids": member_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "var_names": list(VAR_NAMES),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "X": torch.from_numpy(X_all).to(torch_dtype),
        "times": torch.from_numpy(times_all.astype(np.float32)),
        "member_slices": member_slices,
        "grid_n": 512,
        "n_channels": 4,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    meta = {
        "out_path": str(out_path),
        "n_members": len(member_ids),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "total_frames": int(X_all.shape[0]),
        "dtype": dtype,
        "file_size_gb": out_path.stat().st_size / 1e9,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified U-Net dataset .pt")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_ensemble_200",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse/kh_unet_dataset.pt",
    )
    parser.add_argument("--max-members", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    t0 = time.time()
    print(f">>> packing up to {args.max_members} members from {args.data_dir}")
    meta = build_dataset_pt(
        args.data_dir,
        args.out,
        max_members=args.max_members,
        n_val=args.n_val,
        n_test=args.n_test,
        split_seed=args.split_seed,
        dtype=args.dtype,
    )
    print(f">>> wrote {args.out}  ({meta['file_size_gb']:.2f} GB)")
    print(
        f">>> members={meta['n_members']} train={meta['n_train']} "
        f"val={meta['n_val']} test={meta['n_test']} frames={meta['total_frames']}"
    )
    print(f">>> done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
