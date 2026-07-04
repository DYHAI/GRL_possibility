#!/usr/bin/env python3
"""
Kelvin–Helmholtz via 2D compressible Euler DGSEM (Trixi-matched numerics).

Reference: Trixi `elixir_euler_kelvin_helmholtz_instability_amr.jl`
  polydeg=3, LobattoLegendre, Lax–Friedrichs surface flux, periodic [-1,1]².
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from dgsem_euler import DGSEMConfig, DGSEMEuler2D
from euler_fv import _cons_to_prim, _prim_to_cons

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SimConfig:
    ne: int = 32              # elements / side (Trixi initial_refinement_level=5)
    p: int = 3                # polydeg
    t_end: float = 3.0
    cfl: float = 1.3          # Trixi StepsizeCallback
    save_interval: float = 0.05
    v2_amp: float = 0.1
    step_noise_std: float = 0.015
    step_noise_sigma: float = 1.2
    ic_amp_jitter: float = 0.08


def _interface_mask(Y: np.ndarray, width: float = 0.35) -> np.ndarray:
    return (np.abs(Y) < width).astype(np.float64)


def _seed_params(seed: int, cfg: SimConfig) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 2 * np.pi))
    amp = cfg.v2_amp * (1.0 + cfg.ic_amp_jitter * (rng.random() - 0.5))
    return phase, float(amp)


def run_trajectory(traj_id: int, out_dir: Path, cfg: SimConfig | None = None) -> dict:
    cfg = cfg or SimConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    dg_cfg = DGSEMConfig(
        ne=cfg.ne, p=cfg.p, t_end=cfg.t_end, cfl=cfg.cfl, save_interval=cfg.save_interval,
    )
    solver = DGSEMEuler2D(dg_cfg)
    phase, amp = _seed_params(traj_id, cfg)
    U0 = solver.set_kh_ic(phase=phase, amp=amp)
    mask = _interface_mask(solver.Y)
    rng = np.random.default_rng(traj_id + 99_001)

    def step_noise(U: np.ndarray, dt: float) -> np.ndarray:
        if cfg.step_noise_std <= 0:
            return U
        eta = gaussian_filter(rng.standard_normal(U.shape[1:]), sigma=cfg.step_noise_sigma)
        eta = eta * mask
        eta -= eta.mean()
        rho, u, v, p = _cons_to_prim(U)
        drho = cfg.step_noise_std * eta
        du = 0.5 * cfg.step_noise_std * eta
        dv = 0.3 * cfg.step_noise_std * eta
        rho2 = np.maximum(rho + drho, 1e-6)
        u2, v2 = u + du, v + dv
        p2 = np.maximum(p, 1e-6)
        return _prim_to_cons(rho2, u2, v2, p2)

    snaps_rho, snaps_u, snaps_v, times = solver.run(U0, step_noise=step_noise)
    rho_hist, states, next_states = DGSEMEuler2D.stacks_from_snaps(snaps_rho, snaps_u)

    np.savez_compressed(
        out_dir / "trajectory.npz",
        states=states,
        next_states=next_states,
        rho=rho_hist,
        u=np.stack(snaps_u, axis=0),
        v=np.stack(snaps_v, axis=0),
        times=np.array(times, dtype=np.float32),
        phase=np.float32(phase),
        amp=np.float32(amp),
        x=solver.x1d.astype(np.float32),
        y=solver.y1d.astype(np.float32),
    )
    meta = {
        "traj_id": traj_id,
        "seed": traj_id,
        "phase": phase,
        "amp": amp,
        "backend": "python_dgsem_trixi_match",
        "n_nodes": solver.n1d,
        **asdict(cfg),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-id", type=int, default=1)
    parser.add_argument("--ne", type=int, default=32, help="DG elements per side")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = SimConfig(ne=args.ne)
    out = args.out or ROOT / "outputs" / "kelvin_helmholtz" / "ensemble" / f"traj_{args.traj_id:03d}"
    meta = run_trajectory(args.traj_id, out, cfg)
    print(
        f"Saved {out}  ne={cfg.ne}  nodes={meta['n_nodes']}  "
        f"phase={meta['phase']:.3f}  amp={meta['amp']:.4f}"
    )


if __name__ == "__main__":
    main()
