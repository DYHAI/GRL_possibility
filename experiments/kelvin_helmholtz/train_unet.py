#!/usr/bin/env python3
"""Train U-Net one-step forecaster: 512×512×4 -> 512×512×4 with per-pixel RMSE loss."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.models import UNet
from experiments.kelvin_helmholtz.kh_dataset import (
    N_CHANNELS,
    OneStepDataset,
    PtOneStepDataset,
    PtRandomOneStepDataset,
    RandomOneStepDataset,
    compute_norm_stats,
    count_one_step_pairs,
    discover_members,
    load_members,
    load_pt_dataset,
    pt_train_pair_count,
    pt_val_pair_count,
    save_split_json,
    split_members_three_way,
)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x0, x1 in loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x0)
        loss = rmse_loss(pred, x1)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_rmse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x0, x1 in loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        pred = model(x0)
        total += float(rmse_loss(pred, x1).item())
        n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KH U-Net one-step forecaster")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_ensemble_200",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse",
    )
    parser.add_argument("--n-val", type=int, default=10, help="Validation members (early stopping)")
    parser.add_argument("--n-test", type=int, default=20, help="Test members (eval only, never trained)")
    parser.add_argument("--max-members", type=int, default=200, help="Use at most N members (0 = all)")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--random-train", action="store_true", default=True,
                        help="Randomly sample (member, t)->(t+1) pairs each step")
    parser.add_argument("--no-random-train", action="store_false", dest="random_train")
    parser.add_argument("--samples-per-epoch", type=int, default=0,
                        help="Random draws per epoch (0 = all train pairs)")
    parser.add_argument(
        "--dataset-pt",
        type=Path,
        default=None,
        help="Unified .pt dataset (from build_unet_pt.py); skips NPZ loading",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-ch", type=int, default=32, help="U-Net base channels")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dataset_pt is not None and args.dataset_pt.exists():
        print(f">>> loading unified dataset: {args.dataset_pt}")
        pt_data = load_pt_dataset(args.dataset_pt)
        mean = pt_data["mean"].numpy()
        std = pt_data["std"].numpy()
        train_ids = pt_data["train_ids"]
        val_ids = pt_data["val_ids"]
        test_ids = pt_data.get("test_ids", [])
        print(
            f">>> members: total={len(pt_data['member_ids'])} "
            f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
        )
        print(f">>> random train sampling: {args.random_train}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        split_path = args.out_dir / "split.json"
        save_split_json(
            split_path,
            Path(pt_data.get("data_dir", args.data_dir)),
            train_ids,
            val_ids,
            test_ids,
            mean,
            std,
            pt_data.get("split_seed", args.split_seed),
        )

        n_train_pairs = pt_train_pair_count(pt_data)
        n_val_pairs = pt_val_pair_count(pt_data)
        samples_per_epoch = args.samples_per_epoch or n_train_pairs

        if args.random_train:
            train_ds = PtRandomOneStepDataset(
                pt_data,
                samples_per_epoch=samples_per_epoch,
                seed=args.split_seed,
            )
        else:
            train_ds = PtOneStepDataset(pt_data, train_ids)
        val_ds = PtOneStepDataset(pt_data, val_ids)
    else:
        member_ids = discover_members(args.data_dir)
        if args.max_members > 0:
            member_ids = member_ids[: args.max_members]
        n_val = args.n_val
        n_test = args.n_test
        n_min = n_val + n_test + 1
        if len(member_ids) < n_min:
            raise SystemExit(
                f"Need at least {n_min} members with 512x4 NPZ "
                f"(train + {n_val} val + {n_test} test); found {len(member_ids)} in {args.data_dir}"
            )

        train_ids, val_ids, test_ids = split_members_three_way(
            member_ids, n_val=n_val, n_test=n_test, seed=args.split_seed
        )
        print(
            f">>> members: total={len(member_ids)} "
            f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
        )
        print(f">>> random train sampling: {args.random_train}")

        train_members = load_members(args.data_dir, train_ids)
        val_members = load_members(args.data_dir, val_ids)
        mean, std = compute_norm_stats(train_members)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        split_path = args.out_dir / "split.json"
        save_split_json(
            split_path, args.data_dir, train_ids, val_ids, test_ids, mean, std, args.split_seed
        )

        n_train_pairs = count_one_step_pairs(train_members)
        n_val_pairs = count_one_step_pairs(val_members)
        samples_per_epoch = args.samples_per_epoch or n_train_pairs

        if args.random_train:
            train_ds = RandomOneStepDataset(
                train_members, mean, std, samples_per_epoch=samples_per_epoch, seed=args.split_seed
            )
        else:
            train_ds = OneStepDataset(train_members, mean, std)
        val_ds = OneStepDataset(val_members, mean, std)

    print(f">>> pairs: train_pool={n_train_pairs}  samples/epoch={len(train_ds)}  val={n_val_pairs}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet(in_ch=N_CHANNELS, out_ch=N_CHANNELS, base=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val = float("inf")
    best_path = args.out_dir / "unet_best.pt"

    print(f">>> training on {device} for {args.epochs} epochs")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.random_train and hasattr(train_ds, "rng"):
            train_ds.rng = np.random.default_rng(args.split_seed + epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_rmse(model, val_loader, device)
        row = {"epoch": epoch, "train_rmse": train_loss, "val_rmse": val_loss}
        history.append(row)
        print(f"epoch {epoch:3d}  train_rmse={train_loss:.6f}  val_rmse={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "in_ch": N_CHANNELS,
                    "out_ch": N_CHANNELS,
                    "base_ch": args.base_ch,
                    "best_val_rmse": best_val,
                    "epoch": epoch,
                },
                best_path,
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": mean,
            "std": std,
            "in_ch": N_CHANNELS,
            "out_ch": N_CHANNELS,
            "base_ch": args.base_ch,
            "epoch": args.epochs,
        },
        args.out_dir / "unet_last.pt",
    )

    summary = {
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_ch": args.base_ch,
        "best_val_rmse": best_val,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "samples_per_epoch": len(train_ds),
        "random_train": args.random_train,
        "wall_sec": time.time() - t0,
        "history": history,
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n>>> best val RMSE={best_val:.6f}  saved {best_path}")


if __name__ == "__main__":
    main()
