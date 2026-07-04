"""Extreme-value diagnostics for KH fields (ρ, vorticity, gradients)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_IDX = 0
V1_IDX = 1
V2_IDX = 2


@dataclass
class ExtremeThresholds:
    """Quantile thresholds for ρ, learned from training members only."""

    rho_q01: float
    rho_q05: float
    rho_q10: float
    rho_q90: float
    rho_q95: float
    rho_q99: float
    vort_max_q95: float
    grad_rho_max_q95: float

    def as_dict(self) -> dict:
        return {
            "rho_q01": self.rho_q01,
            "rho_q05": self.rho_q05,
            "rho_q10": self.rho_q10,
            "rho_q90": self.rho_q90,
            "rho_q95": self.rho_q95,
            "rho_q99": self.rho_q99,
            "vort_max_q95": self.vort_max_q95,
            "grad_rho_max_q95": self.grad_rho_max_q95,
        }


def vorticity(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dv1_dy, dv1_dx = np.gradient(v1)
    dv2_dy, dv2_dx = np.gradient(v2)
    return dv2_dx - dv1_dy


def grad_magnitude(field: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(field)
    return np.sqrt(gx**2 + gy**2)


def frame_extreme_scalars(x: np.ndarray) -> dict[str, float]:
    """Scalars for one frame x: (4, H, W)."""
    rho = x[RHO_IDX]
    v1 = x[V1_IDX]
    v2 = x[V2_IDX]
    omega = vorticity(v1, v2)
    grad_r = grad_magnitude(rho)
    speed = np.sqrt(v1**2 + v2**2)
    return {
        "rho_max": float(rho.max()),
        "rho_min": float(rho.min()),
        "rho_p99": float(np.quantile(rho, 0.99)),
        "rho_p01": float(np.quantile(rho, 0.01)),
        "vort_max": float(np.max(np.abs(omega))),
        "grad_rho_max": float(grad_r.max()),
        "speed_max": float(speed.max()),
    }


def compute_train_thresholds(train_members: dict) -> ExtremeThresholds:
    rho_vals: list[np.ndarray] = []
    vort_max_vals: list[float] = []
    grad_max_vals: list[float] = []
    for series in train_members.values():
        for t in range(len(series.X)):
            sc = frame_extreme_scalars(series.X[t])
            rho_vals.append(series.X[t, RHO_IDX].ravel())
            vort_max_vals.append(sc["vort_max"])
            grad_max_vals.append(sc["grad_rho_max"])
    pooled = np.concatenate(rho_vals)
    return ExtremeThresholds(
        rho_q01=float(np.quantile(pooled, 0.01)),
        rho_q05=float(np.quantile(pooled, 0.05)),
        rho_q10=float(np.quantile(pooled, 0.10)),
        rho_q90=float(np.quantile(pooled, 0.90)),
        rho_q95=float(np.quantile(pooled, 0.95)),
        rho_q99=float(np.quantile(pooled, 0.99)),
        vort_max_q95=float(np.quantile(vort_max_vals, 0.95)),
        grad_rho_max_q95=float(np.quantile(grad_max_vals, 0.95)),
    )


def exceedance_fraction(rho: np.ndarray, threshold: float, tail: str = "high") -> float:
    if tail == "high":
        return float(np.mean(rho >= threshold))
    return float(np.mean(rho <= threshold))


def conditional_extreme_bias(truth: np.ndarray, pred: np.ndarray, q_high: float, q_low: float) -> dict:
    """Bias on tail pixels: forecast - truth where truth is in top/bottom train quantile."""
    rho_t = truth[RHO_IDX]
    rho_p = pred[RHO_IDX]
    high_mask = rho_t >= q_high
    low_mask = rho_t <= q_low
    out = {}
    if high_mask.any():
        out["high_tail_bias"] = float(np.mean(rho_p[high_mask] - rho_t[high_mask]))
        out["high_tail_peak_ratio"] = float(np.mean(rho_p[high_mask]) / (np.mean(rho_t[high_mask]) + 1e-30))
    if low_mask.any():
        out["low_tail_bias"] = float(np.mean(rho_p[low_mask] - rho_t[low_mask]))
        out["low_tail_peak_ratio"] = float(np.mean(rho_p[low_mask]) / (np.mean(rho_t[low_mask]) + 1e-30))
    return out


def compare_frame_extremes(
    truth: np.ndarray,
    pred: np.ndarray,
    thresholds: ExtremeThresholds,
) -> dict:
    """Compare one truth frame vs prediction (rollout or ensemble mean)."""
    st = frame_extreme_scalars(truth)
    sp = frame_extreme_scalars(pred)
    rho_t = truth[RHO_IDX]
    rho_p = pred[RHO_IDX]

    cond = conditional_extreme_bias(truth, pred, thresholds.rho_q95, thresholds.rho_q05)
    return {
        **{f"truth_{k}": v for k, v in st.items()},
        **{f"pred_{k}": v for k, v in sp.items()},
        "peak_ratio_rho_max": sp["rho_max"] / (st["rho_max"] + 1e-30),
        "peak_ratio_vort_max": sp["vort_max"] / (st["vort_max"] + 1e-30),
        "peak_ratio_grad_rho_max": sp["grad_rho_max"] / (st["grad_rho_max"] + 1e-30),
        "exceed_q99_truth": exceedance_fraction(rho_t, thresholds.rho_q99, "high"),
        "exceed_q99_pred": exceedance_fraction(rho_p, thresholds.rho_q99, "high"),
        "exceed_q01_truth": exceedance_fraction(rho_t, thresholds.rho_q01, "low"),
        "exceed_q01_pred": exceedance_fraction(rho_p, thresholds.rho_q01, "low"),
        "vort_event_truth": float(st["vort_max"] >= thresholds.vort_max_q95),
        "vort_event_pred": float(sp["vort_max"] >= thresholds.vort_max_q95),
        **cond,
    }


def rollout_extreme_series(
    truth_seq: np.ndarray,
    pred_seq: np.ndarray,
    ens_seq: np.ndarray | None,
    thresholds: ExtremeThresholds,
) -> dict:
    """
    Compare rollout trajectories.
    truth_seq, pred_seq: (T, 4, H, W); ens_seq optional same shape.
    """
    T = min(len(truth_seq), len(pred_seq))
    vs_truth = [compare_frame_extremes(truth_seq[t], pred_seq[t], thresholds) for t in range(T)]
    vs_ens = []
    if ens_seq is not None:
        for t in range(T):
            if t < len(ens_seq):
                vs_ens.append(compare_frame_extremes(truth_seq[t], ens_seq[t], thresholds))

    def series(key: str, rows: list[dict]) -> list[float]:
        return [r[key] for r in rows if key in r]

    out = {
        "n_steps": T,
        "peak_ratio_rho_max_by_step": series("peak_ratio_rho_max", vs_truth),
        "peak_ratio_vort_max_by_step": series("peak_ratio_vort_max", vs_truth),
        "high_tail_bias_by_step": series("high_tail_bias", vs_truth),
        "low_tail_bias_by_step": series("low_tail_bias", vs_truth),
        "vort_event_truth_by_step": series("vort_event_truth", vs_truth),
        "vort_event_pred_by_step": series("vort_event_pred", vs_truth),
        "truth_rho_max_by_step": series("truth_rho_max", vs_truth),
        "pred_rho_max_by_step": series("pred_rho_max", vs_truth),
    }
    if vs_ens:
        out["ens_peak_ratio_rho_max_by_step"] = series("peak_ratio_rho_max", vs_ens)
        out["ens_high_tail_bias_by_step"] = series("high_tail_bias", vs_ens)
        out["ens_rho_max_by_step"] = series("pred_rho_max", vs_ens)

    out["mean_peak_ratio_rho_max"] = float(np.mean(out["peak_ratio_rho_max_by_step"]))
    out["min_peak_ratio_rho_max"] = float(np.min(out["peak_ratio_rho_max_by_step"]))
    out["mean_high_tail_bias"] = float(np.mean(out["high_tail_bias_by_step"])) if out["high_tail_bias_by_step"] else None
    out["vort_event_count_truth"] = int(np.sum(out["vort_event_truth_by_step"]))
    out["vort_event_count_pred"] = int(np.sum(out["vort_event_pred_by_step"]))
    return out


def pooled_exceedance_curve(rho_list: list[np.ndarray], thresholds: np.ndarray) -> np.ndarray:
    """Fraction of pooled pixels exceeding each threshold."""
    pooled = np.concatenate([r.ravel() for r in rho_list])
    return np.array([np.mean(pooled >= t) for t in thresholds], dtype=np.float64)
