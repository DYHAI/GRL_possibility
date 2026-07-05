#!/usr/bin/env python3
"""Pack member NPZ files into a single .pt for fast U-Net training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from experiments.kelvin_helmholtz.kh_dataset import (
    N_CHANNELS,
    VAR_NAMES,
    compute_norm_stats,
    discover_members,
    load_members,
    split_members_three_way,
)


def pack_dataset(
    data_dir: Path,
    out_path: Path,
    max_members: int = 200,
    n_val: int = 10,
    n_test: int = 20,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> dict:
    member_ids = discover_members(data_dir)
    if max_members > 0:
        member_ids = member_ids[:max_members]
    train_ids, val_ids, test_ids = split_members_three_way(
        member_ids, n_val=n_val, n_test=n_test, seed=seed
    )

    all_members = load_members(data_dir, member_ids)
    train_members = {mid: all_members[mid] for mid in train_ids}
    mean, std = compute_norm_stats(train_members)

    packed_members: dict[int, dict] = {}
    for mid in member_ids:
        s = all_members[mid]
        packed_members[mid] = {
            "X": torch.from_numpy(s.X).to(dtype),
            "times": torch.from_numpy(s.times).to(torch.float32),
            "n_frames": int(len(s.X)),
        }

    payload = {
        "format": "kh_unet_packed_v1",
        "data_dir": str(data_dir),
        "member_ids": member_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "split_seed": seed,
        "var_names": list(VAR_NAMES),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "members": packed_members,
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
        "dtype": str(dtype).replace("torch.", ""),
        "file_size_gb": out_path.stat().st_size / 1e9,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Pack KH NPZ ensemble into one .pt")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_ensemble_200",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse/kh_dataset_180.pt",
    )
    p.add_argument("--max-members", type=int, default=180)
    p.add_argument("--n-val", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Store X in float16 (half size)")
    args = p.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    print(f">>> packing {args.max_members} members from {args.data_dir}")
    meta = pack_dataset(
        args.data_dir, args.out, args.max_members, args.n_val, args.seed, dtype=dtype
    )
    print(f">>> wrote {args.out}  ({meta['file_size_gb']:.2f} GB)")
    print(f">>> train={meta['n_train']}  val={meta['n_val']}")


if __name__ == "__main__":
    main()
