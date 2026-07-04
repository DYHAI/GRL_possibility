#!/usr/bin/env python3
"""Visualize Trixi VTU snapshots (yellow-blue full-domain panel + GIF)."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

try:
    import meshio
except ImportError as exc:
    raise SystemExit("pip install meshio") from exc

from kh_colormap import KH_CMAP, RHO_VMAX, RHO_VMIN

ROOT = Path(__file__).resolve().parents[2]


def vtu_files(vtu_dir: Path) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(vtu_dir / "solution_*.vtu")))


def read_density(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = meshio.read(path)
    pts = mesh.points[:, :2]
    for key in ("rho", "density", "Density"):
        if key in mesh.point_data:
            return pts[:, 0], pts[:, 1], np.asarray(mesh.point_data[key], dtype=np.float64)
    name = next(iter(mesh.point_data))
    return pts[:, 0], pts[:, 1], np.asarray(mesh.point_data[name], dtype=np.float64)


def to_grid(x, y, rho, nx: int = 256, ny: int = 256) -> np.ndarray:
    xi = np.linspace(-1, 1, nx)
    yi = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), rho, (X, Y), method="linear")
    if np.isnan(Z).any():
        Zn = griddata((x, y), rho, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Zn, Z)
    return Z.astype(np.float32)


def render_frame(rho_grid: np.ndarray, out_png: Path, title: str = "") -> None:
    extent = (-1.0, 1.0, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=160)
    im = ax.imshow(
        rho_grid.T,
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
    if title:
        fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho$")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_gif(png_dir: Path, gif_path: Path, pattern: str = "frame_*.png") -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    frames = sorted(png_dir.glob(pattern))
    if not frames:
        return
    imageio.mimsave(gif_path, [imageio.imread(f) for f in frames], duration=0.25, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vtu-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_10steps/vtu_converted",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_10steps",
    )
    args = parser.parse_args()

    files = vtu_files(args.vtu_dir)
    if not files:
        raise SystemExit(f"No VTU in {args.vtu_dir}")

    png_dir = args.out_dir / "frames"
    png_dir.mkdir(parents=True, exist_ok=True)
    meta = []

    for i, vtu in enumerate(files):
        x, y, rho = read_density(vtu)
        grid = to_grid(x, y, rho)
        png = png_dir / f"frame_{i:04d}.png"
        step = vtu.stem.split("_")[-1]
        render_frame(grid, png, title=f"Trixi KH AMR  step={step}  (t≈{int(step)*0.005:.3f})")
        meta.append({"vtu": vtu.name, "png": png.name, "n_points": len(rho)})
        print(f"  {vtu.name} -> {png.name}")

    gif_path = args.out_dir / "kh_10steps.gif"
    make_gif(png_dir, gif_path)

    # comparison: first vs last
    x0, y0, r0 = read_density(files[0])
    x1, y1, r1 = read_density(files[-1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    for ax, grid, lab in zip(
        axes,
        [to_grid(x0, y0, r0), to_grid(x1, y1, r1)],
        [f"t=0 (step {files[0].stem.split('_')[-1]})", f"t≈0.05 (step {files[-1].stem.split('_')[-1]})"],
    ):
        im = ax.imshow(
            grid.T,
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
    compare_png = args.out_dir / "compare_t0_t10.png"
    fig.tight_layout()
    fig.savefig(compare_png, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_frames": len(files),
        "vtu_dir": str(args.vtu_dir),
        "gif": str(gif_path),
        "compare": str(compare_png),
        "frames": meta,
    }
    (args.out_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nGIF: {gif_path}")
    print(f"Compare: {compare_png}")


if __name__ == "__main__":
    main()
