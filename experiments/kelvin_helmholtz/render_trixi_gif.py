#!/usr/bin/env python3
"""Build single-panel GIF from Trixi snapshots.npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from kh_colormap import KH_CMAP, RHO_VMAX, RHO_VMIN

ROOT = Path(__file__).resolve().parents[2]


def to_grid(cx, cy, rho, n=1024) -> np.ndarray:
    xi = yi = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xi, yi, indexing="ij")
    Z = griddata((cx, cy), rho, (X, Y), method="linear")
    if np.isnan(Z).any():
        Zn = griddata((cx, cy), rho, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Zn, Z)
    return Z.astype(np.float32)


def render_frame(rho_grid: np.ndarray, out: Path, title: str = "") -> None:
    extent = (-1.0, 1.0, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=120)
    ax.imshow(
        rho_grid, origin="lower", extent=extent, cmap=KH_CMAP,
        vmin=RHO_VMIN, vmax=RHO_VMAX, interpolation="bilinear", aspect="equal",
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "outputs/kelvin_helmholtz/trixi_gif_3s")
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--duration-ms", type=int, default=120)
    args = parser.parse_args()

    npz = args.data_dir / "snapshots.npz"
    if not npz.exists():
        raise SystemExit(f"Missing {npz}")

    data = np.load(npz)
    steps = sorted(int(s) for s in data["steps"])
    frames_dir = args.data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    pngs = []
    for i, step in enumerate(steps):
        cx = data[f"cx_{step}"]
        cy = data[f"cy_{step}"]
        rho = data[f"rho_{step}"]
        grid = to_grid(cx, cy, rho, n=args.grid_n)
        png = frames_dir / f"frame_{i:04d}.png"
        t_guess = step * (3.0 / max(steps[-1], 1))
        render_frame(grid, png, title=f"step {step}  ($t \\approx {t_guess:.2f}$)")
        pngs.append(png)
        print(f"  {step} -> {png.name}")

    gif = args.data_dir / "kh_3s.gif"
    imageio.mimsave(gif, [imageio.imread(p) for p in pngs],
                    duration=args.duration_ms / 1000.0, loop=0)

    # best frame still PNG
    best = steps[len(steps) * 3 // 4]  # late-time structure
    grid = to_grid(data[f"cx_{best}"], data[f"cy_{best}"], data[f"rho_{best}"], n=args.grid_n)
    hd = args.data_dir / "hd" / f"kh_step{best:04d}.png"
    hd.parent.mkdir(parents=True, exist_ok=True)
    render_frame(grid, hd, title=f"step {best} (late)")

    summary = {"steps": steps, "n_frames": len(pngs), "gif": str(gif), "hd": str(hd)}
    (args.data_dir / "gif_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nGIF: {gif}")
    print(f"HD:  {hd}")


if __name__ == "__main__":
    main()
