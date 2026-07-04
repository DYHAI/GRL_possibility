#!/usr/bin/env python3
"""Plot Trixi 10-step snapshots from snapshots.npz."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from kh_colormap import KH_CMAP, RHO_VMAX, RHO_VMIN

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "kelvin_helmholtz" / "trixi_10steps"


def to_grid(cx, cy, rho, nx=256, ny=256) -> np.ndarray:
    """Return rho[iy, ix] on a uniform grid (x horizontal, y vertical)."""
    xi = np.linspace(-1, 1, nx)
    yi = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(xi, yi, indexing="ij")
    Z = griddata((cx, cy), rho, (X, Y), method="linear")
    if np.isnan(Z).any():
        Zn = griddata((cx, cy), rho, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Zn, Z)
    return Z.astype(np.float32)


def render_frame(rho_grid: np.ndarray, out: Path, title: str) -> None:
    extent = (-1.0, 1.0, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=160)
    im = ax.imshow(
        rho_grid,
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
    fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho$")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir

    npz_path = out / "snapshots.npz"
    if not npz_path.exists():
        raise SystemExit(f"Missing {npz_path} — run trixi/export_npz.jl first")

    data = np.load(npz_path)
    steps = sorted(int(s) for s in data["steps"])
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    for i, step in enumerate(steps):
        cx = data[f"cx_{step}"]
        cy = data[f"cy_{step}"]
        rho = data[f"rho_{step}"]
        grid = to_grid(cx, cy, rho)
        png = frames_dir / f"frame_{i:04d}.png"
        render_frame(grid, png, title=f"Trixi KH AMR  step={step}")
        meta.append({"step": step, "png": png.name})
        print(f"  step {step} -> {png.name}")

    # compare first vs last
    g0 = to_grid(data[f"cx_{steps[0]}"], data[f"cy_{steps[0]}"], data[f"rho_{steps[0]}"])
    g1 = to_grid(data[f"cx_{steps[-1]}"], data[f"cy_{steps[-1]}"], data[f"rho_{steps[-1]}"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    for ax, grid, lab in zip(axes, [g0, g1], [f"step {steps[0]}", f"step {steps[-1]}"]):
        im = ax.imshow(
            grid,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap=KH_CMAP,
            vmin=RHO_VMIN,
            vmax=RHO_VMAX,
            aspect="equal",
        )
        ax.set_title(lab)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label=r"$\rho$")
    compare = out / "compare_t0_tend.png"
    fig.tight_layout()
    fig.savefig(compare, bbox_inches="tight")
    plt.close(fig)

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    pngs = sorted(frames_dir.glob("frame_*.png"))
    gif = out / "kh_evolution.gif"
    if pngs:
        imageio.mimsave(gif, [imageio.imread(p) for p in pngs], duration=0.3, loop=0)

    summary = {"steps": steps, "compare": str(compare), "gif": str(gif), "frames": meta}
    (out / "viz_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nCompare: {compare}")
    print(f"GIF: {gif}")


if __name__ == "__main__":
    main()
