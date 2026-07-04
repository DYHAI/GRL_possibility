"""
2D compressible Euler finite-volume solver (periodic box).

Matches Trixi elixir_euler_kelvin_helmholtz_instability*.jl:
  - gamma = 1.4, domain [-1,1]^2, IC from Rueda-Ramírez & Gassner (2021)
  - HLL Riemann flux (closer to Trixi LLF but less smearing than pure Rusanov on fine features)
  - SSP-RK3, CFL-limited dt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAMMA = 1.4


def _prim_to_cons(rho, u, v, p):
    ke = 0.5 * rho * (u * u + v * v)
    E = p / (GAMMA - 1.0) + ke
    return np.stack([rho, rho * u, rho * v, E], axis=0)


def _cons_to_prim(U: np.ndarray) -> tuple[np.ndarray, ...]:
    rho = np.maximum(U[0], 1e-8)
    u = U[1] / rho
    v = U[2] / rho
    ke = 0.5 * rho * (u * u + v * v)
    p = np.maximum((GAMMA - 1.0) * (U[3] - ke), 1e-8)
    return rho, u, v, p


def _flux_x(U: np.ndarray) -> np.ndarray:
    rho, u, v, p = _cons_to_prim(U)
    return np.stack(
        [rho * u, rho * u * u + p, rho * u * v, (U[3] + p) * u],
        axis=0,
    )


def _flux_y(U: np.ndarray) -> np.ndarray:
    rho, u, v, p = _cons_to_prim(U)
    return np.stack(
        [rho * v, rho * u * v, rho * v * v + p, (U[3] + p) * v],
        axis=0,
    )


def _max_wave_speed(U: np.ndarray) -> np.ndarray:
    rho, u, v, p = _cons_to_prim(U)
    c = np.sqrt(GAMMA * p / rho)
    return np.abs(u) + c, np.abs(v) + c


def kh_initial_primitives(
    X: np.ndarray, Y: np.ndarray, *, phase: float = 0.0, amp: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same IC as Trixi `initial_condition_kelvin_helmholtz_instability`."""
    slope = 15.0
    B = np.tanh(slope * Y + 7.5) - np.tanh(slope * Y - 7.5)
    rho = 0.5 + 0.75 * B
    u = 0.5 * (B - 1.0)
    v = amp * np.sin(2.0 * np.pi * X + phase)
    p = np.ones_like(rho)
    return rho.astype(np.float64), u, v, p


@dataclass
class EulerFVConfig:
    nx: int = 256
    ny: int = 256
    t_end: float = 3.0
    cfl: float = 0.45
    save_interval: float = 0.05
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0


def _hll_flux(Fn, UL: np.ndarray, UR: np.ndarray, sL: np.ndarray, sR: np.ndarray) -> np.ndarray:
    """HLL flux; sL/sR are scalar fields (most negative / most positive wave speed)."""
    FL, FR = Fn(UL), Fn(UR)
    out = np.empty_like(UL)
    zero = sL >= 0
    out[..., zero] = FL[..., zero]
    pos = sR <= 0
    out[..., pos] = FR[..., pos]
    mid = ~(zero | pos)
    if mid.any():
        denom = sR - sL
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        for c in range(UL.shape[0]):
            out[c, mid] = (
                sR[mid] * FL[c, mid] - sL[mid] * FR[c, mid]
                + sL[mid] * sR[mid] * (UR[c, mid] - UL[c, mid])
            ) / denom[mid]
    return out


class EulerFV2D:
    def __init__(self, cfg: EulerFVConfig | None = None):
        self.cfg = cfg or EulerFVConfig()
        c = self.cfg
        self.dx = (c.x_max - c.x_min) / c.nx
        self.dy = (c.y_max - c.y_min) / c.ny
        x = c.x_min + (np.arange(c.nx) + 0.5) * self.dx
        y = c.y_min + (np.arange(c.ny) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

    def set_kh_ic(self, *, phase: float = 0.0, amp: float = 0.1) -> np.ndarray:
        rho, u, v, p = kh_initial_primitives(self.X, self.Y, phase=phase, amp=amp)
        return _prim_to_cons(rho, u, v, p).astype(np.float64)

    def _hll_flux_x(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        rhoL, uL, _, pL = _cons_to_prim(UL)
        rhoR, uR, _, pR = _cons_to_prim(UR)
        cL = np.sqrt(GAMMA * pL / rhoL)
        cR = np.sqrt(GAMMA * pR / rhoR)
        sL = np.minimum(uL - cL, uR - cR)
        sR = np.maximum(uL + cL, uR + cR)
        return _hll_flux(_flux_x, UL, UR, sL, sR)

    def _hll_flux_y(self, DL: np.ndarray, DR: np.ndarray) -> np.ndarray:
        rhoL, _, vL, pL = _cons_to_prim(DL)
        rhoR, _, vR, pR = _cons_to_prim(DR)
        cL = np.sqrt(GAMMA * pL / rhoL)
        cR = np.sqrt(GAMMA * pR / rhoR)
        sL = np.minimum(vL - cL, vR - cR)
        sR = np.maximum(vL + cL, vR + cR)
        return _hll_flux(_flux_y, DL, DR, sL, sR)

    def _rhs(self, U: np.ndarray) -> np.ndarray:
        UR = np.roll(U, -1, axis=1)
        DR = np.roll(U, -1, axis=2)
        Fx = self._hll_flux_x(U, UR)
        Fy = self._hll_flux_y(U, DR)
        dU = -(Fx - np.roll(Fx, 1, axis=1)) / self.dx
        dU -= (Fy - np.roll(Fy, 1, axis=2)) / self.dy
        return dU

    def _dt(self, U: np.ndarray) -> float:
        sx, sy = _max_wave_speed(U)
        smax = max(float(sx.max()), float(sy.max()), 1e-6)
        return self.cfg.cfl * min(self.dx, self.dy) / smax

    def _rk3_step(self, U: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._rhs(U)
        U1 = U + dt * k1
        k2 = self._rhs(U1)
        U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
        k3 = self._rhs(U2)
        Un = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * k3)
        rho, _, _, p = _cons_to_prim(Un)
        bad = (rho <= 0) | (p <= 0) | ~np.isfinite(Un).all(axis=0)
        if bad.any():
            raise RuntimeError(f"non-physical state: {int(bad.sum())} cells")
        return Un

    def run(
        self,
        U0: np.ndarray,
        *,
        step_noise: callable | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
        U = U0.copy()
        t = 0.0
        next_save = 0.0
        rho, u, v, _ = _cons_to_prim(U)
        snaps_rho = [rho.astype(np.float32)]
        snaps_u = [u.astype(np.float32)]
        snaps_v = [v.astype(np.float32)]
        times = [0.0]

        while t < self.cfg.t_end - 1e-12:
            dt = min(self._dt(U), self.cfg.t_end - t)
            U = self._rk3_step(U, dt)
            if step_noise is not None:
                U = step_noise(U, dt)
            t += dt
            if t >= next_save - 1e-10:
                rho, u, v, _ = _cons_to_prim(U)
                snaps_rho.append(rho.astype(np.float32))
                snaps_u.append(u.astype(np.float32))
                snaps_v.append(v.astype(np.float32))
                times.append(t)
                next_save += self.cfg.save_interval

        rho, u, v, _ = _cons_to_prim(U)
        if times[-1] < self.cfg.t_end - 1e-6:
            snaps_rho.append(rho.astype(np.float32))
            snaps_u.append(u.astype(np.float32))
            snaps_v.append(v.astype(np.float32))
            times.append(self.cfg.t_end)

        return snaps_rho, snaps_u, snaps_v, times

    @staticmethod
    def stacks_from_snaps(
        snaps_rho: list[np.ndarray],
        snaps_u: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.stack(snaps_rho, axis=0)
        u = np.stack(snaps_u, axis=0)
        states = np.stack([rho[:-1], u[:-1]], axis=1)
        next_states = np.stack([rho[1:], u[1:]], axis=1)
        return rho, states, next_states
