#!/usr/bin/env python3
"""Generate KH ensemble trajectories in parallel (pure Python)."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from simulate import SimConfig, run_trajectory

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "kelvin_helmholtz" / "ensemble"


def _worker(traj_id: int, cfg_dict: dict) -> dict:
    cfg = SimConfig(**cfg_dict)
    out_dir = OUT / f"traj_{traj_id:03d}"
    t0 = time.perf_counter()
    meta = run_trajectory(traj_id, out_dir, cfg)
    meta["wall_sec"] = time.perf_counter() - t0
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--ne", type=int, default=32, help="DG elements per side (Trixi ref level 5)")
    args = parser.parse_args()

    cfg = SimConfig(ne=args.ne)
    cfg_dict = cfg.__dict__
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Python KH ensemble: {args.n_traj} traj, {args.workers} workers")
    t0 = time.perf_counter()
    results = []
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, i, cfg_dict): i for i in range(1, args.n_traj + 1)}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                meta = fut.result()
                results.append(meta)
                print(f"[OK] traj {tid:03d}  {meta['wall_sec']:.1f}s")
            except Exception as exc:
                failed += 1
                print(f"[FAIL] traj {tid:03d}  {exc}")

    total = time.perf_counter() - t0
    summary = {
        "n_traj": args.n_traj,
        "workers": args.workers,
        "failed": failed,
        "total_wall_sec": round(total, 1),
        "backend": "python",
    }
    if results:
        summary["mean_traj_sec"] = round(sum(r["wall_sec"] for r in results) / len(results), 2)
    (OUT / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {total:.1f}s ({total/60:.1f} min). Failed: {failed}")


if __name__ == "__main__":
    main()
