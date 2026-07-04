#!/usr/bin/env python3
"""
Render KH density in Trixi-style yellow–blue full-domain panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kh_colormap import KH_CMAP, RHO_VMAX, RHO_VMIN, upsample_field

ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE = ROOT / "outputs" / "kelvin_helmholtz" / "ensemble"
OUT = ROOT / "outputs" / "kelvin_helmholtz" / "style_v2"


def _best_frame_index(rho: np.ndarray, times: np.ndarray) -> int:
    """Pick time when interface distortion is strongest (before full mix)."""
    ny = rho.shape[1]
    y = np.linspace(-1, 1, ny, endpoint=False)
    band = np.abs(y) < 0.45
    scores = []
    for k in range(len(times)):
        grad_y = np.gradient(rho[k], axis=0)
        scores.append(float(np.abs(grad_y[band, :]).mean()))
    # prefer mid-late instability, not t=0
    idx = int(np.argmax(scores[max(1, len(scores) // 10) :]) + max(1, len(scores) // 10))
    return min(idx, len(times) - 1)


def render_panel(
    rho: np.ndarray,
    *,
    title: str = "",
    upsample: int = 2,
) -> plt.Figure:
    field = upsample_field(rho, upsample)
    extent = (-1.0, 1.0, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=200)
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=extent,
        cmap=KH_CMAP,
        vmin=RHO_VMIN,
        vmax=RHO_VMAX,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Density $\rho$", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def render_trajectory(traj_id: int, *, upsample: int = 2, make_gif: bool = True) -> None:
    path = ENSEMBLE / f"traj_{traj_id:03d}" / "trajectory.npz"
    d = np.load(path)
    rho, times = d["rho"], d["times"]
    OUT.mkdir(parents=True, exist_ok=True)

    fi = _best_frame_index(rho, times)
    fig = render_panel(
        rho[fi],
        title=f"KH Traj {traj_id}  $t={times[fi]:.2f}$  (Trixi IC, Euler FV)",
        upsample=upsample,
    )
    png = OUT / f"trajectory_{traj_id:02d}.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png}")

    if not make_gif:
        return

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    frames = []
    # GIF: subsample times, focus on instability window
    indices = np.linspace(0, len(times) - 1, min(120, len(times)), dtype=int)
    for i in indices:
        fig = render_panel(rho[i], upsample=max(1, upsample // 2))
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(img.copy())
        plt.close(fig)

    gif_path = OUT / f"trajectory_{traj_id:02d}.gif"
    imageio.mimsave(gif_path, frames, duration=0.08, loop=0)
    print(f"Saved {gif_path}")


def render_four_grid(traj_ids: list[int], upsample: int = 2) -> None:
    """4×6 grid like paper draft but yellow–blue density."""
    paths = [ENSEMBLE / f"traj_{t:03d}" / "trajectory.npz" for t in traj_ids]
    rhos, times_list = [], []
    for p in paths:
        d = np.load(p)
        rhos.append(d["rho"])
        times_list.append(d["times"])

    n_traj = len(traj_ids)
    frame_idx = np.linspace(0, rhos[0].shape[0] - 1, 6, dtype=int)

    fig, axes = plt.subplots(n_traj, 6, figsize=(14, 2.4 * n_traj), dpi=200)
    if n_traj == 1:
        axes = axes[np.newaxis, :]

    for row, (tid, rho, times) in enumerate(zip(traj_ids, rhos, times_list)):
        for col, fi in enumerate(frame_idx):
            ax = axes[row, col]
            field = upsample_field(rho[fi], upsample)
            ax.imshow(
                field.T,
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap=KH_CMAP,
                vmin=RHO_VMIN,
                vmax=RHO_VMAX,
                interpolation="bilinear",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$t={times[fi]:.2f}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Traj {tid}", fontsize=10)

    fig.suptitle("KH density — yellow/blue (Trixi-style)", fontsize=12)
    fig.tight_layout()
    out = OUT / "four_trajectories_yellow_blue.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-ids", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--upsample", type=int, default=2)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--grid-only", action="store_true")
    args = parser.parse_args()

    if args.grid_only:
        render_four_grid(args.traj_ids, upsample=args.upsample)
        return

    for tid in args.traj_ids:
        render_trajectory(tid, upsample=args.upsample, make_gif=not args.no_gif)
    render_four_grid(args.traj_ids, upsample=args.upsample)


if __name__ == "__main__":
    main()
