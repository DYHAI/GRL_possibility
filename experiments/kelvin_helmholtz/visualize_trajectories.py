#!/usr/bin/env python3
"""Visualize KH ensemble trajectories (static grid + GIF)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE = ROOT / "outputs" / "kelvin_helmholtz" / "ensemble"
FIG = ROOT / "outputs" / "kelvin_helmholtz" / "figures"


def vorticity(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    dvdx = np.gradient(v, dx, axis=0)
    dudy = np.gradient(u, dx, axis=1)
    return (dvdx - dudy).astype(np.float32)


def load_traj(traj_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = ENSEMBLE / f"traj_{traj_id:03d}" / "trajectory.npz"
    d = np.load(path)
    rho, u = d["rho"], d["u"]
    v = d["v"] if "v" in d else np.zeros_like(rho)
    times = d["times"]
    nx = rho.shape[-1]
    dx = 2.0 / nx
    omega = np.stack([vorticity(u[i], v[i], dx) for i in range(len(rho))])
    return rho, u, v, omega, times


def pick_frames(n: int, n_show: int = 6) -> np.ndarray:
    idx = np.linspace(0, n - 1, n_show, dtype=int)
    return np.unique(idx)


def plot_four_trajectories(
    traj_ids: list[int],
    frame_idx: list[int] | None = None,
    field: str = "vorticity",
    out_png: Path | None = None,
) -> None:
    n_traj = len(traj_ids)
    rho0, _, _, _, times = load_traj(traj_ids[0])
    if frame_idx is None:
        frame_idx = pick_frames(len(rho0), 6).tolist()

    fig, axes = plt.subplots(
        n_traj, len(frame_idx),
        figsize=(2.4 * len(frame_idx), 2.4 * n_traj),
        squeeze=False,
    )

    for row, tid in enumerate(traj_ids):
        rho, u, v, omega, times = load_traj(tid)
        for col, fi in enumerate(frame_idx):
            ax = axes[row, col]
            if field == "rho":
                data = rho[fi]
                vmin, vmax = 0.25, 1.75
                cmap = "RdYlBu_r"
                clabel = r"$\rho$"
            else:
                data = omega[fi]
                lim = np.percentile(np.abs(omega), 99)
                vmin, vmax = -lim, lim
                cmap = "RdBu_r"
                clabel = r"$\omega$"
            im = ax.imshow(
                data.T, origin="lower", extent=(-1, 1, -1, 1),
                cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$t={times[fi]:.2f}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Traj {tid}", fontsize=10)

    fig.subplots_adjust(right=0.92, wspace=0.05, hspace=0.08)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(clabel, fontsize=10)
    title = "KH vorticity (Euler FV, Trixi IC)" if field == "vorticity" else "KH density"
    fig.suptitle(title, fontsize=12, y=1.01)

    suffix = "vorticity" if field == "vorticity" else "density"
    out_png = out_png or FIG / f"four_trajectories_{suffix}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def make_gif(traj_id: int, out_path: Path | None = None, fps: int = 10) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    rho, _, _, _, times = load_traj(traj_id)
    vmin, vmax = 0.25, 1.75
    frames = []
    for i in range(len(rho)):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(rho[i].T, origin="lower", extent=(-1, 1, -1, 1),
                  cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"Traj {traj_id}  t={times[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(img.copy())
        plt.close(fig)

    out_path = out_path or FIG / f"trajectory_{traj_id:03d}.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-ids", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--gif", action="store_true", help="also write one GIF per traj")
    parser.add_argument("--field", choices=("vorticity", "rho"), default="vorticity")
    args = parser.parse_args()

    plot_four_trajectories(args.traj_ids, field=args.field)
    plot_four_trajectories(args.traj_ids, field="rho")
    if args.gif:
        for tid in args.traj_ids:
            make_gif(tid)


if __name__ == "__main__":
    main()
