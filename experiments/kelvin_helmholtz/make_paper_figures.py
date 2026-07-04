#!/usr/bin/env python3
"""Generate paper figures from U-Net eval outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.kelvin_helmholtz.kh_dataset import load_member, load_split_json
from experiments.kelvin_helmholtz.eval_unet import load_model, rollout
from experiments.kelvin_helmholtz.kh_colormap import KH_CMAP, RHO_VMAX, RHO_VMIN


def plot_rho_panel(rho: np.ndarray, title: str, ax, vmin=RHO_VMIN, vmax=RHO_VMAX) -> None:
    ax.imshow(rho, origin="lower", cmap=KH_CMAP, vmin=vmin, vmax=vmax, extent=[-1, 1, -1, 1])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper figures: KH U-Net rollout")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_ensemble_200",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse/unet_best.pt",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse/split.json",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse/eval/eval_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/figures",
    )
    parser.add_argument("--member-id", type=int, default=None, help="Test member to visualize")
    parser.add_argument("--step", type=int, default=-1, help="Rollout step index for snapshot")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split_json(args.split_json)
    test_ids = split["test_ids"]
    member_id = args.member_id if args.member_id is not None else test_ids[0]

    model, mean, std = load_model(args.checkpoint, device)
    series = load_member(args.data_dir, member_id)
    rolled = rollout(model, series.X[0], len(series.X) - 1, mean, std, device)
    step = args.step if args.step >= 0 else len(series.X) - 1
    step = min(step, len(series.X) - 1)

    truth = series.X[step, 0]
    forecast = rolled[step, 0]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_rho_panel(truth, f"truth ρ  member {member_id:03d}  step {step}", axes[0])
    plot_rho_panel(forecast, f"U-Net rollout ρ  step {step}", axes[1])
    fig.suptitle("Kelvin–Helmholtz: truth vs RMSE-optimal one-step rollout")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"rollout_rho_member_{member_id:03d}_step_{step:02d}.png", dpi=180)
    plt.close(fig)

    if args.eval_json.exists():
        summary = json.loads(args.eval_json.read_text())
        members = summary.get("members", [])
        if members:
            fig, ax = plt.subplots(figsize=(7, 4))
            for m in members:
                ratios = m["high_k_power_ratio_by_step"]
                ax.plot(np.arange(1, len(ratios) + 1), ratios, alpha=0.5, lw=1)
            mean_ratio = np.mean([m["high_k_power_ratio_by_step"] for m in members], axis=0)
            ax.plot(np.arange(1, len(mean_ratio) + 1), mean_ratio, "k-", lw=2.5, label="test mean")
            ax.axhline(1.0, color="gray", ls="--")
            ax.set_xlabel("rollout step")
            ax.set_ylabel("high-k power ratio (forecast / truth)")
            ax.set_title("Long rollout: high-wavenumber spectral energy (ρ)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.out_dir / "high_k_ratio_all_test_members.png", dpi=180)
            plt.close(fig)

    print(f">>> figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
