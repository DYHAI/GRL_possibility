"""
2D nodal DGSEM for compressible Euler — mirrors Trixi KH elixir settings.

Trixi reference (elixir_euler_kelvin_helmholtz_instability_amr.jl):
  - CompressibleEulerEquations2D, gamma=1.4
  - LobattoLegendreBasis polydeg=3
  - FluxLaxFriedrichs(max_abs_speed_naive)
  - TreeMesh initial_refinement_level=5 → 32×32 elements on [-1,1]²
  - boundary_condition_periodic, tspan (0, 3), CFL ≈ 1.3

Duplicate DOF at element interfaces (discontinuous storage) so Lax–Friedrichs
surface fluxes provide the needed numerical dissipation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from euler_fv import (
    _cons_to_prim,
    _flux_x,
    _flux_y,
    _max_wave_speed,
    _prim_to_cons,
    kh_initial_primitives,
)


def _gll_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n == 4:
        a = 1.0 / np.sqrt(5.0)
        x = np.array([-1.0, -a, a, 1.0])
        w = np.array([1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0])
        return x, w
    N = n - 1
    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    x[0], x[-1] = -1.0, 1.0
    for _ in range(20):
        P = np.polynomial.legendre.legvander(x, N)[-1]
        dP = N * (x * P - np.polynomial.legendre.legvander(x, N - 1)[-1]) / (x * x - 1.0 + 1e-15)
        x = x - (1.0 - x * x) * dP / (N * (N + 1) * P + 1e-15)
        x[0], x[-1] = -1.0, 1.0
    w = 2.0 / (N * (N + 1) * np.polynomial.legendre.legvander(x, N)[-1] ** 2)
    return x, w


def _dmat(r: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = len(r)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (r[i] - r[j])
        D[i, i] = -np.sum(D[i, :])
    return D


def _lf_flux_x(UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
    sxL, _ = _max_wave_speed(UL)
    sxR, _ = _max_wave_speed(UR)
    s = np.maximum(sxL, sxR)
    FL, FR = _flux_x(UL), _flux_x(UR)
    return 0.5 * (FL + FR) - 0.5 * s * (UR - UL)


def _lf_flux_y(DL: np.ndarray, DR: np.ndarray) -> np.ndarray:
    _, syL = _max_wave_speed(DL)
    _, syR = _max_wave_speed(DR)
    s = np.maximum(syL, syR)
    FL, FR = _flux_y(DL), _flux_y(DR)
    return 0.5 * (FL + FR) - 0.5 * s * (DR - DL)


@dataclass
class DGSEMConfig:
    ne: int = 32
    p: int = 3
    t_end: float = 3.0
    cfl: float = 0.8
    save_interval: float = 0.05
    filter_eta: float = 36.0  # exponential modal filter (Trixi-style stabilization)


class DGSEMEuler2D:
    """Nodal DGSEM (duplicate interface DOF) + Lax–Friedrichs surface fluxes."""

    def __init__(self, cfg: DGSEMConfig | None = None):
        self.cfg = cfg or DGSEMConfig()
        c = self.cfg
        self.p = c.p
        self.ne = c.ne
        self.np = c.p + 1
        self.n1d = c.ne * self.np
        self.h = 2.0 / c.ne

        self.r, self.w = _gll_nodes_weights(self.np)
        self.D = _dmat(self.r, self.w)
        self._filter_mat = self._build_filter(c.filter_eta)
        self.x1d, self.y1d = self._global_coords()
        self.X, self.Y = np.meshgrid(self.x1d, self.y1d, indexing="ij")

    def _build_filter(self, eta: float) -> np.ndarray:
        """Modal exponential filter on GLL nodes (per element)."""
        p = self.p
        v = np.polynomial.legendre.legvander(self.r, p)
        modal = np.linalg.inv(v)
        s = np.exp(-eta * (np.arange(p + 1) / max(p, 1)) ** 4)
        return v @ (s[:, None] * modal)

    def _global_coords(self) -> tuple[np.ndarray, np.ndarray]:
        ne, np_, r, h = self.ne, self.np, self.r, self.h
        n = ne * np_
        x = np.empty(n)
        for e in range(ne):
            x0, x1 = -1.0 + e * h, -1.0 + (e + 1) * h
            x[e * np_ : (e + 1) * np_] = 0.5 * ((x1 - x0) * r + (x1 + x0))
        return x, x.copy()

    def set_kh_ic(self, *, phase: float = 0.0, amp: float = 0.1) -> np.ndarray:
        rho, u, v, p = kh_initial_primitives(self.X, self.Y, phase=phase, amp=amp)
        return _prim_to_cons(rho, u, v, p)

    def _dir_dg(
        self,
        F: np.ndarray,
        U: np.ndarray,
        lf_fn,
        flux_fn,
        axis: int,
    ) -> np.ndarray:
        """Strong-form DG derivative along axis (0=x, 1=y)."""
        ne, p, np_, D, h = self.ne, self.p, self.np, self.D, self.h
        nvar = F.shape[0]
        invh = 2.0 / h
        dU = np.zeros_like(U)

        def _apply(Fb: np.ndarray, Ub: np.ndarray, dUb: np.ndarray) -> None:
            for e in range(ne):
                base = e * np_
                sl = slice(base, base + np_)
                blk = Fb[:, sl, :]
                for m in range(nvar):
                    dUb[m, sl, :] -= invh * (D @ blk[m])
            for e in range(ne):
                base = e * np_
                e_prev = (e - 1) % ne
                e_next = (e + 1) % ne
                iL = base
                iR = base + p
                iR_prev = e_prev * np_ + p
                iL_next = e_next * np_
                f_left = lf_fn(Ub[:, iR_prev, :], Ub[:, iL, :])
                f_right = lf_fn(Ub[:, iR, :], Ub[:, iL_next, :])
                Fl = flux_fn(Ub[:, iL, :])
                Fr = flux_fn(Ub[:, iR, :])
                for m in range(nvar):
                    dUb[m, iL, :] -= invh * (f_left[m] - Fl[m])
                    dUb[m, iR, :] += invh * (f_right[m] - Fr[m])

        if axis == 0:
            _apply(F, U, dU)
        else:
            Ut = np.transpose(U, (0, 2, 1))
            Ft = np.transpose(F, (0, 2, 1))
            dUt = np.zeros_like(Ut)
            _apply(Ft, Ut, dUt)
            dU = np.transpose(dUt, (0, 2, 1))
        return dU

    def rhs(self, U: np.ndarray) -> np.ndarray:
        Fx, Fy = _flux_x(U), _flux_y(U)
        return self._dir_dg(Fx, U, _lf_flux_x, _flux_x, 0) + self._dir_dg(
            Fy, U, _lf_flux_y, _flux_y, 1
        )

    def _dt(self, U: np.ndarray) -> float:
        sx, sy = _max_wave_speed(U)
        smax = max(float(sx.max()), float(sy.max()), 1e-6)
        dx_min = self.h / self.p
        return self.cfg.cfl * dx_min / smax

    def _apply_filter(self, U: np.ndarray) -> np.ndarray:
        """Filter each element's tensor-product patch (primitive variables)."""
        if self.cfg.filter_eta <= 0:
            return U
        ne, np_, F = self.ne, self.np, self._filter_mat
        rho, u, v, p = _cons_to_prim(U)
        for name, arr in (("rho", rho), ("u", u), ("v", v), ("p", p)):
            for e in range(ne):
                xs = slice(e * np_, e * np_ + np_)
                for ey in range(ne):
                    ys = slice(ey * np_, ey * np_ + np_)
                    blk = arr[xs, ys]
                    for j in range(np_):
                        blk[:, j] = F @ blk[:, j]
                    for i in range(np_):
                        blk[i, :] = F @ blk[i, :]
                    arr[xs, ys] = blk
        return _prim_to_cons(rho, u, v, p)

    def _rk3_step(self, U: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(U)
        U1 = self._apply_filter(U + dt * k1)
        k2 = self.rhs(U1)
        U2 = self._apply_filter(0.75 * U + 0.25 * (U1 + dt * k2))
        k3 = self.rhs(U2)
        Un = self._apply_filter((1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * k3))
        rho, _, _, p = _cons_to_prim(Un)
        if (rho <= 0).any() or (p <= 0).any() or not np.isfinite(Un).all():
            raise RuntimeError("DGSEM: non-physical state")
        return Un

    def _collapse_x(self, field: np.ndarray) -> np.ndarray:
        """Drop duplicate interface DOF along axis 0 → ne*p+1 unique nodes."""
        ne, p, np_ = self.ne, self.p, self.np
        chunks = [field[e * np_ : e * np_ + p] for e in range(ne)]
        chunks.append(field[(ne - 1) * np_ + p : (ne - 1) * np_ + p + 1])
        return np.concatenate(chunks, axis=0)

    def _collapse(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho_c = self._collapse_x(self._collapse_x(rho.T).T)
        u_c = self._collapse_x(self._collapse_x(u.T).T)
        v_c = self._collapse_x(self._collapse_x(v.T).T)
        return rho_c, u_c, v_c

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
        rho, u, v = self._collapse(rho, u, v)
        sr, su, sv, times = [rho.astype(np.float32)], [u.astype(np.float32)], [v.astype(np.float32)], [0.0]

        while t < self.cfg.t_end - 1e-12:
            dt = min(self._dt(U), self.cfg.t_end - t)
            U = self._rk3_step(U, dt)
            if step_noise is not None:
                U = step_noise(U, dt)
            t += dt
            if t >= next_save - 1e-10:
                rho, u, v, _ = _cons_to_prim(U)
                rho, u, v = self._collapse(rho, u, v)
                sr.append(rho.astype(np.float32))
                su.append(u.astype(np.float32))
                sv.append(v.astype(np.float32))
                times.append(t)
                next_save += self.cfg.save_interval

        rho, u, v, _ = _cons_to_prim(U)
        rho, u, v = self._collapse(rho, u, v)
        if times[-1] < self.cfg.t_end - 1e-6:
            sr.append(rho.astype(np.float32))
            su.append(u.astype(np.float32))
            sv.append(v.astype(np.float32))
            times.append(self.cfg.t_end)
        return sr, su, sv, times

    @staticmethod
    def stacks_from_snaps(snaps_rho, snaps_u):
        rho = np.stack(snaps_rho, axis=0)
        u = np.stack(snaps_u, axis=0)
        states = np.stack([rho[:-1], u[:-1]], axis=1)
        next_states = np.stack([rho[1:], u[1:]], axis=1)
        return rho, states, next_states
