#!/usr/bin/env python3
"""
Convert ensemble cell data -> regular 512×512×4 grids for U-Net training.

Per-member mode (default, align_frames=0 in Julia export):
  member_001/member_001_512x4.npz  X: (T_m, 4, H, W), times, steps

Aligned mode (KH_EXPORT_ALIGN=1):
  unet_training_512.npz  X: (n_members, T, 4, H, W)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
TRIXI = ROOT / "experiments/kelvin_helmholtz/trixi"
VARS = ("rho", "v1", "v2", "p")


def to_grid(cx: np.ndarray, cy: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    xi = yi = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(xi, yi, indexing="ij")
    Z = griddata((cx, cy), vals, (X, Y), method="linear")
    if np.isnan(Z).any():
        Zn = griddata((cx, cy), vals, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Zn, Z)
    return Z.astype(np.float32)


def run_julia_export(data_dir: Path, nvis: int, julia: Path) -> Path:
    raw = data_dir / "ensemble_fields_4var.npz"
    env = {**dict(__import__("os").environ), "NVIS": str(nvis)}
    subprocess.run(
        [str(julia), str(TRIXI / "export_ensemble_4var.jl"), str(data_dir)],
        check=True,
        env=env,
    )
    if not raw.exists():
        raise SystemExit(f"Julia export did not create {raw}")
    return raw


def build_member_grid(raw: np.lib.npyio.NpzFile, member_id: int, grid_n: int) -> dict:
    nf = int(raw[f"member_{member_id}_nframes"])
    times = np.asarray(raw[f"member_{member_id}_times"], dtype=np.float32)
    steps = np.zeros(nf, dtype=np.int32)
    X = np.zeros((nf, len(VARS), grid_n, grid_n), dtype=np.float32)
    for fi in range(nf):
        if f"step_{member_id}_{fi}" in raw:
            steps[fi] = int(raw[f"step_{member_id}_{fi}"])
        cx = raw[f"cx_{member_id}_{fi}"]
        cy = raw[f"cy_{member_id}_{fi}"]
        for ci, var in enumerate(VARS):
            X[fi, ci] = to_grid(cx, cy, raw[f"{var}_{member_id}_{fi}"], n=grid_n)
    return {
        "X": X,
        "times": times,
        "steps": steps,
        "n_frames": np.int32(nf),
        "member_id": np.int32(member_id),
        "grid_n": np.int32(grid_n),
        "n_channels": np.int32(len(VARS)),
        "var_names": np.array(VARS),
    }


def export_one_member(raw_path: Path, data_dir: Path, member_id: int, grid_n: int) -> Path:
    raw = np.load(raw_path)
    member_dir = data_dir / f"member_{member_id:03d}"
    member_dir.mkdir(parents=True, exist_ok=True)
    arrays = build_member_grid(raw, member_id, grid_n)
    out_npz = member_dir / f"member_{member_id:03d}_{grid_n}x4.npz"
    np.savez_compressed(out_npz, **arrays)
    meta = {
        "member_id": member_id,
        "n_frames": int(arrays["n_frames"]),
        "shape": list(arrays["X"].shape),
        "t_first": float(arrays["times"][0]),
        "t_last": float(arrays["times"][-1]),
        "times": arrays["times"].tolist(),
        "file": str(out_npz.relative_to(data_dir)),
    }
    (member_dir / f"member_{member_id:03d}_{grid_n}x4.json").write_text(json.dumps(meta, indent=2))
    print(
        f"  member {member_id}: {meta['n_frames']} frames  "
        f"t=[{meta['t_first']:.3f}, {meta['t_last']:.3f}]  -> {out_npz.name}",
        flush=True,
    )
    return out_npz


def export_per_member(raw_path: Path, data_dir: Path, grid_n: int) -> list[Path]:
    raw = np.load(raw_path)
    align = int(raw["align_frames"]) if "align_frames" in raw else 1
    if align != 0:
        raise SystemExit(
            "Per-member grid export requires align_frames=0 (KH_EXPORT_ALIGN=0). "
            "Re-run Julia export or use --combined for aligned stack."
        )

    members = [int(m) for m in raw["members"]]
    out_paths: list[Path] = []
    summary = {
        "grid_n": grid_n,
        "var_names": list(VARS),
        "save_dt": float(raw["save_dt"]) if "save_dt" in raw else None,
        "members": {},
    }

    for m in members:
        member_dir = data_dir / f"member_{m:03d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        arrays = build_member_grid(raw, m, grid_n)
        out_npz = member_dir / f"member_{m:03d}_{grid_n}x4.npz"
        np.savez_compressed(out_npz, **arrays)
        meta = {
            "member_id": m,
            "n_frames": int(arrays["n_frames"]),
            "shape": list(arrays["X"].shape),
            "t_first": float(arrays["times"][0]),
            "t_last": float(arrays["times"][-1]),
            "times": arrays["times"].tolist(),
            "file": str(out_npz.relative_to(data_dir)),
        }
        (member_dir / f"member_{m:03d}_{grid_n}x4.json").write_text(json.dumps(meta, indent=2))
        summary["members"][str(m)] = meta
        out_paths.append(out_npz)
        print(
            f"  member {m}: {meta['n_frames']} frames  "
            f"t=[{meta['t_first']:.3f}, {meta['t_last']:.3f}]  -> {out_npz.name}"
        )

    summary_path = data_dir / f"unet_grids_{grid_n}x4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n>>> Summary: {summary_path}")
    return out_paths


def build_unet_arrays_aligned(raw: np.lib.npyio.NpzFile, grid_n: int) -> tuple[dict, dict]:
    members = [int(m) for m in raw["members"]]
    n_frames = int(raw["n_frames"])
    n_members = len(members)

    X = np.zeros((n_members, n_frames, len(VARS), grid_n, grid_n), dtype=np.float32)
    steps = np.zeros(n_frames, dtype=np.int32)
    times = np.zeros(n_frames, dtype=np.float32)

    if "step_0" in raw:
        for fi in range(n_frames):
            steps[fi] = int(raw[f"step_{fi}"])
            if f"time_{fi}" in raw:
                times[fi] = float(raw[f"time_{fi}"])

    for mi, m in enumerate(members):
        for fi in range(n_frames):
            cx = raw[f"cx_{m}_{fi}"]
            cy = raw[f"cy_{m}_{fi}"]
            for ci, var in enumerate(VARS):
                X[mi, fi, ci] = to_grid(cx, cy, raw[f"{var}_{m}_{fi}"], n=grid_n)
        print(f"  member {m}: {n_frames} frames -> ({len(VARS)}, {grid_n}, {grid_n})")

    X_flat = X.reshape(n_members * n_frames, len(VARS), grid_n, grid_n)
    mean = X_flat.mean(axis=(0, 2, 3))
    std = X_flat.std(axis=(0, 2, 3))
    std = np.where(std < 1e-8, 1.0, std)

    meta = {
        "grid_n": grid_n,
        "n_members": n_members,
        "n_frames": n_frames,
        "var_names": list(VARS),
        "shape": list(X.shape),
        "flat_shape": list(X_flat.shape),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "members": members,
        "steps": steps.tolist(),
        "times": times.tolist(),
    }
    if "t0" in raw:
        meta["t0"] = float(raw["t0"])
    if "t_end" in raw:
        meta["t_end"] = float(raw["t_end"])

    arrays = {
        "X": X,
        "X_flat": X_flat,
        "members": np.array(members, dtype=np.int32),
        "n_members": np.int32(n_members),
        "n_frames": np.int32(n_frames),
        "steps": steps,
        "times": times,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "var_names": np.array(VARS),
        "grid_n": np.int32(grid_n),
    }
    return arrays, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 512×512×4 U-Net training arrays")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_ensemble_branch",
    )
    parser.add_argument("--grid-n", type=int, default=512)
    parser.add_argument("--nvis", type=int, default=8, help="VTU subcell resolution (Julia export)")
    parser.add_argument(
        "--julia",
        type=Path,
        default=ROOT / "tools/julia-1.11.9/bin/julia",
    )
    parser.add_argument("--skip-julia", action="store_true", help="Use existing ensemble_fields_4var.npz")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Write single unet_training_512.npz (aligned mode only)",
    )
    parser.add_argument("--member-id", type=int, default=None, help="Export one member only")
    parser.add_argument(
        "--fields-npz",
        type=Path,
        default=None,
        help="Per-member fields NPZ (default: member_XXX/member_XXX_fields_4var.npz)",
    )
    args = parser.parse_args()

    if args.member_id is not None:
        mid = args.member_id
        raw_path = args.fields_npz or (
            args.data_dir / f"member_{mid:03d}" / f"member_{mid:03d}_fields_4var.npz"
        )
        if not raw_path.exists():
            raise SystemExit(f"Missing {raw_path}; run Julia export_one_member_4var.jl first")
        print(f">>> Building U-Net grid  member={mid}  grid={args.grid_n}", flush=True)
        export_one_member(raw_path, args.data_dir, mid, args.grid_n)
        return

    raw_path = args.data_dir / "ensemble_fields_4var.npz"
    if args.skip_julia:
        if not raw_path.exists():
            raise SystemExit("Missing ensemble_fields_4var.npz; run without --skip-julia")
    else:
        raw_path = run_julia_export(data_dir=args.data_dir, nvis=args.nvis, julia=args.julia)

    print(f">>> Building U-Net grids  grid={args.grid_n}  dir={args.data_dir}")

    if args.combined:
        raw = np.load(raw_path)
        arrays, meta = build_unet_arrays_aligned(raw, args.grid_n)
        out_npz = args.data_dir / "unet_training_512.npz"
        np.savez_compressed(out_npz, **arrays)
        meta_path = args.data_dir / "unet_training_512.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\n>>> Wrote {out_npz}")
        print(f"    X shape: {arrays['X'].shape}")
        return

    export_per_member(raw_path, args.data_dir, args.grid_n)


if __name__ == "__main__":
    main()
