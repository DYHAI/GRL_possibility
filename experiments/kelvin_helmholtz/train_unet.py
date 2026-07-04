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
    compute_norm_stats,
    discover_members,
    load_members,
    save_split_json,
    split_members,
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
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=42)
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

    member_ids = discover_members(args.data_dir)
    if len(member_ids) < args.n_test + 1:
        raise SystemExit(
            f"Need at least {args.n_test + 1} members with 512x4 NPZ; found {len(member_ids)} in {args.data_dir}"
        )

    train_ids, test_ids = split_members(member_ids, n_test=args.n_test, seed=args.split_seed)
    print(f">>> members: total={len(member_ids)} train={len(train_ids)} test={len(test_ids)}")

    train_members = load_members(args.data_dir, train_ids)
    test_members = load_members(args.data_dir, test_ids)
    mean, std = compute_norm_stats(train_members)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / "split.json"
    save_split_json(split_path, args.data_dir, train_ids, test_ids, mean, std, args.split_seed)

    train_ds = OneStepDataset(train_members, mean, std)
    test_ds = OneStepDataset(test_members, mean, std)
    print(f">>> pairs: train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet(in_ch=N_CHANNELS, out_ch=N_CHANNELS, base=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_test = float("inf")
    best_path = args.out_dir / "unet_best.pt"

    print(f">>> training on {device} for {args.epochs} epochs")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss = eval_rmse(model, test_loader, device)
        row = {"epoch": epoch, "train_rmse": train_loss, "test_rmse": test_loss}
        history.append(row)
        print(f"epoch {epoch:3d}  train_rmse={train_loss:.6f}  test_rmse={test_loss:.6f}")
        if test_loss < best_test:
            best_test = test_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "in_ch": N_CHANNELS,
                    "out_ch": N_CHANNELS,
                    "base_ch": args.base_ch,
                    "best_test_rmse": best_test,
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
        "best_test_rmse": best_test,
        "train_pairs": len(train_ds),
        "test_pairs": len(test_ds),
        "wall_sec": time.time() - t0,
        "history": history,
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n>>> best test RMSE={best_test:.6f}  saved {best_path}")


if __name__ == "__main__":
    main()
