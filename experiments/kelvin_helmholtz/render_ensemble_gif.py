#!/usr/bin/env python3
"""Render 5-member branch ensemble as one GIF (1×N panel grid)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from kh_colormap import KH_CMAP, RHO_VMAX, RHO_VMIN

ROOT = Path(__file__).resolve().parents[2]


def to_grid(cx, cy, rho, n=512) -> np.ndarray:
    xi = yi = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xi, yi, indexing="ij")
    Z = griddata((cx, cy), rho, (X, Y), method="linear")
    if np.isnan(Z).any():
        Zn = griddata((cx, cy), rho, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Zn, Z)
    return Z.astype(np.float32)


def render_ensemble_frame(
    grids: list[np.ndarray],
    out: Path,
    *,
    title: str = "",
    labels: list[str] | None = None,
) -> None:
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.4), dpi=120)
    if n == 1:
        axes = [axes]
    extent = (-1.0, 1.0, -1.0, 1.0)
    for ax, grid in zip(axes, grids):
        ax.imshow(
            grid,
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
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.90, bottom=0.02)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def iter_gif_frame_times(data: np.lib.npyio.NpzFile, members: list[int]):
    """Yield (frame_label_time, frame_index_per_member) for GIF rendering."""
    align = int(data["align_frames"]) if "align_frames" in data else 1
    if align != 0:
        n_frames = int(data["n_frames"])
        times = data["times"] if "times" in data else None
        t0 = float(data["t0"]) if "t0" in data else 0.0
        t_end = float(data["t_end"]) if "t_end" in data else 5.0
        for fi in range(n_frames):
            if times is not None:
                t = float(times[fi])
            else:
                t = t0 + (t_end - t0) * fi / max(n_frames - 1, 1)
            yield t, {m: fi for m in members}
        return

    # per-member NPZ: GIF uses times where all members have a snapshot
    save_dt = float(data["save_dt"]) if "save_dt" in data else None
    tol = max(1e-3, (save_dt or 0.1) * 0.05)
    member_times = {m: np.asarray(data[f"member_{m}_times"], dtype=np.float64) for m in members}

    if save_dt is not None:
        t0 = float(data["t0"]) if "t0" in data else float(min(member_times[m][0] for m in members))
        t_max = min(float(member_times[m][-1]) for m in members)
        ti = 0
        while True:
            t = t0 + ti * save_dt
            if t > t_max + tol:
                break
            idx = {}
            ok = True
            for m in members:
                hits = np.where(np.abs(member_times[m] - t) <= tol)[0]
                if len(hits) == 0:
                    ok = False
                    break
                idx[m] = int(hits[0])
            if ok:
                yield t, idx
            ti += 1
        return

    n_common = min(int(data[f"member_{m}_nframes"]) for m in members)
    for fi in range(n_common):
        t = float(member_times[members[0]][fi])
        yield t, {m: fi for m in members}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_ensemble_branch",
    )
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--duration-ms", type=int, default=120)
    args = parser.parse_args()

    for candidate in ("ensemble_snapshots.npz", "ensemble_fields_4var.npz"):
        npz_path = args.data_dir / candidate
        if npz_path.exists():
            break
    else:
        raise SystemExit(f"Missing ensemble NPZ in {args.data_dir}")

    data = np.load(npz_path)
    members = [int(m) for m in data["members"]]
    t0 = float(data["t0"]) if "t0" in data else 0.0
    t_end = float(data["t_end"]) if "t_end" in data else 5.0
    align = int(data["align_frames"]) if "align_frames" in data else 1

    frames_dir = args.data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    pngs = []
    gif_frames = list(iter_gif_frame_times(data, members))
    n_frames = len(gif_frames)
    for gi, (t_guess, idx_map) in enumerate(gif_frames):
        grids = []
        for m in members:
            fi = idx_map[m]
            rho = data[f"rho_{m}_{fi}"]
            cx = data[f"cx_{m}_{fi}"]
            cy = data[f"cy_{m}_{fi}"]
            grids.append(to_grid(cx, cy, rho, n=args.grid_n))
        png = frames_dir / f"frame_{gi:04d}.png"
        mode_note = "" if align else "  [GIF: common times only]"
        render_ensemble_frame(
            grids,
            png,
            title=rf"$t \approx {t_guess:.2f}$  (members 1–5, $t={t0:.1f}\to{t_end:.1f}$){mode_note}",
        )
        pngs.append(png)
        print(f"  frame {gi}  t≈{t_guess:.2f}")

    gif = args.data_dir / "kh_ensemble_branch.gif"
    imageio.mimsave(gif, [imageio.imread(p) for p in pngs],
                    duration=args.duration_ms / 1000.0, loop=0)

    summary = {
        "members": members,
        "n_frames": n_frames,
        "align_frames": align,
        "t0": t0,
        "t_end": t_end,
        "gif": str(gif),
    }
    if align == 0:
        summary["member_nframes"] = {
            str(m): int(data[f"member_{m}_nframes"]) for m in members
        }
    (args.data_dir / "gif_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nGIF: {gif}")


if __name__ == "__main__":
    main()
