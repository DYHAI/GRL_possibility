#!/usr/bin/env python3
"""
HD Trixi KH figure — yellow/blue full-domain density field.

Matches reference: high-res DGSEM+AMR field, horizontal shear layer, fine vortices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from kh_colormap import KH_CMAP, RHO_VMAX, RHO_VMIN

ROOT = Path(__file__).resolve().parents[2]


def to_grid(cx, cy, rho, nx=1024, ny=1024) -> np.ndarray:
    """High-res grid for zoom crops (linear + nearest fill)."""
    from scipy.interpolate import griddata

    xi = np.linspace(-1, 1, nx)
    yi = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(xi, yi, indexing="ij")
    Z = griddata((cx, cy), rho, (X, Y), method="linear")
    if np.isnan(Z).any():
        Zn = griddata((cx, cy), rho, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Zn, Z)
    return Z.astype(np.float32)


def _best_step(data: np.lib.npyio.NpzFile, steps: list[int]) -> int:
    """Pick frame with strongest interface gradient (roll-up)."""
    if len(steps) <= 1:
        return steps[-1]
    scores = []
    for s in steps[1:]:
        rho = data[f"rho_{s}"]
        cy = data[f"cy_{s}"]
        order = np.argsort(cy)
        rho_s = rho[order]
        g = np.abs(np.gradient(rho_s)).mean()
        scores.append(float(g))
    idx = int(np.argmax(scores)) + 1
    return steps[min(idx, len(steps) - 1)]


def render_hd(
    cx: np.ndarray,
    cy: np.ndarray,
    rho: np.ndarray,
    out: Path,
    *,
    title: str = "",
    dpi: int = 300,
    grid_n: int = 1536,
) -> None:
    """Single full-domain panel."""
    Z = to_grid(cx, cy, rho, nx=grid_n, ny=grid_n)
    extent = (-1.0, 1.0, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=dpi)
    im = ax.imshow(
        Z,
        origin="lower",
        extent=extent,
        cmap=KH_CMAP,
        vmin=RHO_VMIN,
        vmax=RHO_VMAX,
        interpolation="bilinear",
        aspect="equal",
        rasterized=True,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=13, y=0.98)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.92)
    cbar.set_label(r"$\rho$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.94, bottom=0.02)
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_t1p0",
    )
    parser.add_argument("--step", type=int, default=None, help="VTK step index; default=best")
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    npz = args.data_dir / "snapshots.npz"
    if not npz.exists():
        raise SystemExit(f"Missing {npz}. Run trixi/export_npz.jl with NVIS>=16 first.")

    data = np.load(npz)
    steps = sorted(int(s) for s in data["steps"])
    step = args.step if args.step is not None else _best_step(data, steps)

    cx = data[f"cx_{step}"]
    cy = data[f"cy_{step}"]
    rho = data[f"rho_{step}"]
    t_approx = step * 0.00318  # ~dt*steps for this setup

    out_dir = args.data_dir / "hd"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"kh_hd_step{step:04d}.png"

    render_hd(
        cx, cy, rho, png,
        title=f"Trixi KH AMR  step={step}  ($t \\approx {t_approx:.2f}$)",
        grid_n=args.grid_n,
        dpi=args.dpi,
    )
    print(f"Saved {png}  ({len(rho)} cells, grid={args.grid_n})")

    meta = {"step": step, "n_cells": int(len(rho)), "grid_n": args.grid_n, "png": str(png)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
