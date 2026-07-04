#!/usr/bin/env python3
"""Convert Trixi VTK snapshots to PNG frames and a GIF."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import meshio
except ImportError as exc:
    raise SystemExit("Install meshio: pip install meshio") from exc


def vtu_files(vtu_dir: Path) -> list[Path]:
    files = sorted(glob.glob(str(vtu_dir / "solution_*.vtu")))
    return [Path(f) for f in files]


def read_density(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = meshio.read(path)
    pts = mesh.points[:, :2]
    rho = None
    for key in ("rho", "density", "Density"):
        if key in mesh.point_data:
            rho = np.asarray(mesh.point_data[key], dtype=np.float64)
            break
    if rho is None:
        names = list(mesh.point_data.keys())
        if not names:
            raise KeyError(f"No point data in {path}")
        rho = np.asarray(mesh.point_data[names[0]], dtype=np.float64)
    return pts[:, 0], pts[:, 1], rho


def render_frame(x, y, rho, out_png: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    sc = ax.scatter(x, y, c=rho, s=1.2, cmap="RdYlBu_r", vmin=0.25, vmax=1.75)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho$")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_gif(png_dir: Path, gif_path: Path, duration_ms: int = 120) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    frames = sorted(png_dir.glob("frame_*.png"))
    if not frames:
        return
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(gif_path, images, duration=duration_ms / 1000.0, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vtu-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs/kelvin_helmholtz/trixi/vtu",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs/kelvin_helmholtz/trixi",
    )
    args = parser.parse_args()

    vtu_dir = args.vtu_dir
    out_dir = args.out_dir
    png_dir = out_dir / "frames"
    png_dir.mkdir(parents=True, exist_ok=True)

    files = vtu_files(vtu_dir)
    if not files:
        raise SystemExit(f"No VTU files found in {vtu_dir}")

    meta = []
    for i, vtu in enumerate(files):
        x, y, rho = read_density(vtu)
        png = png_dir / f"frame_{i:04d}.png"
        render_frame(x, y, rho, png, title=vtu.stem)
        meta.append({"vtu": vtu.name, "png": png.name, "n_points": int(len(rho))})

    gif_path = out_dir / "kh_amr_density.gif"
    make_gif(png_dir, gif_path)

    summary = {"n_frames": len(files), "vtu_dir": str(vtu_dir), "gif": str(gif_path), "frames": meta}
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Exported {len(files)} frames -> {png_dir}")
    print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
