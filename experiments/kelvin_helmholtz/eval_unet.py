#!/usr/bin/env python3
"""
Evaluate U-Net: one-step RMSE, long rollout, radial power spectra (high-k focus),
compare forecast vs truth member vs ensemble mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.models import UNet, denormalize_channels, normalize_channels, radial_energy
from experiments.kelvin_helmholtz.extreme_metrics import (
    compute_train_thresholds,
    pooled_exceedance_curve,
    rollout_extreme_series,
)
from experiments.kelvin_helmholtz.kh_dataset import (
    N_CHANNELS,
    VAR_NAMES,
    discover_members,
    ensemble_mean_at_frame,
    load_member,
    load_members,
    load_split_json,
)


def load_model(checkpoint: Path, device: torch.device) -> tuple[UNet, np.ndarray, np.ndarray]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    base = int(ckpt.get("base_ch", 32))
    model = UNet(in_ch=N_CHANNELS, out_ch=N_CHANNELS, base=base)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    return model, mean, std


@torch.no_grad()
def predict_step(
    model: UNet,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    xn = normalize_channels(x, mean, std)
    xt = torch.from_numpy(xn).unsqueeze(0).to(device)
    yn = model(xt).cpu().numpy()[0]
    return denormalize_channels(yn, mean, std)


def rollout(
    model: UNet,
    x0: np.ndarray,
    n_steps: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Autoregressive rollout returning (n_steps+1, C, H, W)."""
    out = [x0.astype(np.float32)]
    state = x0
    for _ in range(n_steps):
        state = predict_step(model, state, mean, std, device)
        out.append(state)
    return np.stack(out, axis=0)


def field_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rho_power_spectrum(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return radial_energy(rho.astype(np.float64))


def high_k_mask(k: np.ndarray, fraction: float = 1.0 / 3.0) -> np.ndarray:
    """Upper fraction of wavenumber bins (high-k tail)."""
    kmax = k.max()
    cutoff = kmax * (1.0 - fraction)
    return k >= cutoff


def spectrum_metrics(
    truth_rho: np.ndarray,
    pred_rho: np.ndarray,
    high_k_frac: float = 1.0 / 3.0,
) -> dict:
    k_t, p_t = rho_power_spectrum(truth_rho)
    k_p, p_p = rho_power_spectrum(pred_rho)
    hk = high_k_mask(k_t, high_k_frac)
    full_ratio = float(p_p.sum() / (p_t.sum() + 1e-30))
    high_ratio = float(p_p[hk].sum() / (p_t[hk].sum() + 1e-30))
    return {
        "k": k_t.tolist(),
        "power_truth": p_t.tolist(),
        "power_pred": p_p.tolist(),
        "power_ratio_full": full_ratio,
        "power_ratio_high_k": high_ratio,
    }


def evaluate_member(
    model: UNet,
    series,
    all_members: dict,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    high_k_frac: float,
    thresholds,
) -> dict:
    X = series.X
    T = len(X)
    n_roll = T - 1
    rolled = rollout(model, X[0], n_roll, mean, std, device)

    ens_frames = []
    for t in range(T):
        ens = ensemble_mean_at_frame(all_members, t)
        ens_frames.append(ens if ens is not None else np.full_like(X[0], np.nan))
    ens_seq = np.stack(ens_frames, axis=0)

    extreme = rollout_extreme_series(X, rolled, ens_seq, thresholds)

    one_step_rmses = [field_rmse(predict_step(model, X[t], mean, std, device), X[t + 1]) for t in range(n_roll)]

    vs_truth = [field_rmse(rolled[t + 1], X[t + 1]) for t in range(n_roll)]
    vs_ens = []
    for t in range(n_roll):
        ens = ensemble_mean_at_frame(all_members, t + 1)
        if ens is None:
            continue
        vs_ens.append(field_rmse(rolled[t + 1], ens))

    ens_vs_truth = []
    for t in range(n_roll):
        ens = ensemble_mean_at_frame(all_members, t + 1)
        if ens is None:
            continue
        ens_vs_truth.append(field_rmse(ens, X[t + 1]))

    spec_by_step = []
    for t in range(1, T):
        spec_by_step.append(
            spectrum_metrics(X[t, 0], rolled[t, 0], high_k_frac=high_k_frac)
        )

    return {
        "member_id": series.member_id,
        "n_frames": T,
        "one_step_rmse_mean": float(np.mean(one_step_rmses)),
        "rollout_rmse_vs_truth_mean": float(np.mean(vs_truth)),
        "rollout_rmse_vs_truth_final": float(vs_truth[-1]) if vs_truth else None,
        "rollout_rmse_vs_ensemble_mean": float(np.mean(vs_ens)) if vs_ens else None,
        "ensemble_rmse_vs_truth_mean": float(np.mean(ens_vs_truth)) if ens_vs_truth else None,
        "spectrum_by_step": spec_by_step,
        "high_k_power_ratio_by_step": [s["power_ratio_high_k"] for s in spec_by_step],
        "extremes": extreme,
        "rolled": rolled,
    }


def plot_extreme_peaks(member_result: dict, out_path: Path, member_id: int) -> None:
    ex = member_result.get("extremes", {})
    if not ex:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    steps = np.arange(len(ex["truth_rho_max_by_step"]))
    axes[0].plot(steps, ex["truth_rho_max_by_step"], "k-", label="truth member", lw=2)
    axes[0].plot(steps, ex["pred_rho_max_by_step"], "r--", label="U-Net rollout", lw=2)
    if ex.get("ens_rho_max_by_step"):
        axes[0].plot(steps, ex["ens_rho_max_by_step"], "b:", label="ensemble mean", lw=2)
    axes[0].set_xlabel("rollout step")
    axes[0].set_ylabel("domain max ρ")
    axes[0].set_title(f"member {member_id:03d}: peak amplitude vs step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    pr = ex["peak_ratio_rho_max_by_step"]
    axes[1].plot(steps, pr, "o-", ms=3, color="#C44E52")
    if ex.get("ens_peak_ratio_rho_max_by_step"):
        axes[1].plot(steps, ex["ens_peak_ratio_rho_max_by_step"], "s--", ms=3, color="#4C72B0", label="ens. mean")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("rollout step")
    axes[1].set_ylabel("peak ratio (forecast / truth)")
    axes[1].set_title("extreme amplitude underestimation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_exceedance_summary(
    truth_rho: list[np.ndarray],
    pred_rho: list[np.ndarray],
    ens_rho: list[np.ndarray],
    thresholds: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(thresholds, pooled_exceedance_curve(truth_rho, thresholds), "k-", lw=2, label="truth members")
    ax.semilogy(thresholds, pooled_exceedance_curve(pred_rho, thresholds), "r--", lw=2, label="U-Net rollout")
    if ens_rho:
        ax.semilogy(thresholds, pooled_exceedance_curve(ens_rho, thresholds), "b:", lw=2, label="ensemble mean")
    ax.set_xlabel("ρ threshold")
    ax.set_ylabel("exceedance probability P(ρ ≥ threshold)")
    ax.set_title("Extreme-event tail: pooled test rollout")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rollout_spectra(
    member_result: dict,
    out_path: Path,
    member_id: int,
) -> None:
    steps = member_result["spectrum_by_step"]
    if not steps:
        return
    k = np.asarray(steps[0]["k"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Snapshot: last available step
    last = steps[-1]
    axes[0].loglog(k, last["power_truth"], "k-", label="truth member ρ", lw=2)
    axes[0].loglog(k, last["power_pred"], "r--", label="rollout forecast ρ", lw=2)
    axes[0].set_xlabel("wavenumber k")
    axes[0].set_ylabel("radial power")
    axes[0].set_title(f"member {member_id:03d} spectrum at final rollout step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ratios = member_result["high_k_power_ratio_by_step"]
    axes[1].plot(np.arange(1, len(ratios) + 1), ratios, "o-", ms=3)
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("rollout step")
    axes[1].set_ylabel("high-k power ratio (forecast / truth)")
    axes[1].set_title("high-k spectral energy ratio vs rollout step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_bar(summary: dict, out_path: Path) -> None:
    keys = [
        ("one_step_rmse_mean", "one-step RMSE"),
        ("rollout_rmse_vs_truth_mean", "rollout vs truth"),
        ("rollout_rmse_vs_ensemble_mean", "rollout vs ensemble mean"),
        ("ensemble_rmse_vs_truth_mean", "ensemble mean vs truth"),
    ]
    labels, vals = [], []
    for key, label in keys:
        v = summary.get(f"test_{key}_mean")
        if v is not None:
            labels.append(label)
            vals.append(v)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color=["#4C72B0", "#DD8452", "#55A868", "#8172B2"][: len(vals)])
    ax.set_ylabel("RMSE (all channels, pixels)")
    ax.set_title("Test-set mean metrics")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KH U-Net rollout and spectra")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/trixi_ensemble_200",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse/unet_best.pt",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse/split.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/kelvin_helmholtz/unet_mse/eval",
    )
    parser.add_argument(
        "--high-k-frac",
        type=float,
        default=1.0 / 3.0,
        help="Top fraction of k bins treated as high-k (default upper third)",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    split = load_split_json(args.split_json)
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]
    model, mean, std = load_model(args.checkpoint, device)

    all_ids = discover_members(args.data_dir)
    all_members = load_members(args.data_dir, all_ids)
    train_members = load_members(args.data_dir, train_ids)
    test_members = {mid: load_member(args.data_dir, mid) for mid in test_ids}
    thresholds = compute_train_thresholds(train_members)

    print(f">>> eval {len(test_ids)} test members, ensemble pool={len(all_members)}")
    print(f">>> train thresholds: rho_q99={thresholds.rho_q99:.4f} vort_q95={thresholds.vort_max_q95:.4f}")

    pooled_truth_rho: list[np.ndarray] = []
    pooled_pred_rho: list[np.ndarray] = []
    pooled_ens_rho: list[np.ndarray] = []

    member_results = []
    for mid in test_ids:
        print(f"  member {mid:03d} ...", flush=True)
        res = evaluate_member(
            model,
            test_members[mid],
            all_members,
            mean,
            std,
            device,
            args.high_k_frac,
            thresholds,
        )
        member_results.append(res)
        plot_rollout_spectra(res, args.out_dir / "spectra" / f"member_{mid:03d}_spectrum.png", mid)
        plot_extreme_peaks(res, args.out_dir / "extremes" / f"member_{mid:03d}_peaks.png", mid)

        X = test_members[mid].X
        rolled = res["rolled"]
        for t in range(1, len(X)):
            pooled_truth_rho.append(X[t, 0])
            pooled_pred_rho.append(rolled[t, 0])
            ens = ensemble_mean_at_frame(all_members, t)
            if ens is not None:
                pooled_ens_rho.append(ens[0])

    thr_grid = np.linspace(thresholds.rho_q90, thresholds.rho_q99 * 1.02, 40)
    plot_exceedance_summary(
        pooled_truth_rho,
        pooled_pred_rho,
        pooled_ens_rho,
        thr_grid,
        args.out_dir / "extremes" / "exceedance_curve.png",
    )

    def avg_key(key: str) -> float:
        vals = [r[key] for r in member_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "n_test": len(test_ids),
        "n_ensemble_pool": len(all_members),
        "high_k_frac": args.high_k_frac,
        "var_names": list(VAR_NAMES),
        "test_one_step_rmse_mean": avg_key("one_step_rmse_mean"),
        "test_rollout_rmse_vs_truth_mean": avg_key("rollout_rmse_vs_truth_mean"),
        "test_rollout_rmse_vs_truth_final": avg_key("rollout_rmse_vs_truth_final"),
        "test_rollout_rmse_vs_ensemble_mean": avg_key("rollout_rmse_vs_ensemble_mean"),
        "test_ensemble_rmse_vs_truth_mean": avg_key("ensemble_rmse_vs_truth_mean"),
        "test_high_k_power_ratio_final_mean": float(
            np.mean([r["high_k_power_ratio_by_step"][-1] for r in member_results if r["high_k_power_ratio_by_step"]])
        ),
        "extreme_thresholds": thresholds.as_dict(),
        "test_mean_peak_ratio_rho_max": float(
            np.mean([r["extremes"]["mean_peak_ratio_rho_max"] for r in member_results])
        ),
        "test_min_peak_ratio_rho_max": float(
            np.mean([r["extremes"]["min_peak_ratio_rho_max"] for r in member_results])
        ),
        "test_mean_high_tail_bias": float(
            np.nanmean([r["extremes"]["mean_high_tail_bias"] for r in member_results])
        ),
        "test_vort_event_count_truth": int(
            np.sum([r["extremes"]["vort_event_count_truth"] for r in member_results])
        ),
        "test_vort_event_count_pred": int(
            np.sum([r["extremes"]["vort_event_count_pred"] for r in member_results])
        ),
        "members": member_results,
    }
    # rolled arrays are large; omit from JSON
    for r in summary["members"]:
        r.pop("rolled", None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    plot_summary_bar(summary, args.out_dir / "metrics_summary.png")

    print("\n>>> test summary")
    print(f"  one-step RMSE:           {summary['test_one_step_rmse_mean']:.6f}")
    print(f"  rollout vs truth:        {summary['test_rollout_rmse_vs_truth_mean']:.6f}")
    print(f"  rollout vs ens. mean:    {summary['test_rollout_rmse_vs_ensemble_mean']:.6f}")
    print(f"  ensemble mean vs truth:  {summary['test_ensemble_rmse_vs_truth_mean']:.6f}")
    print(f"  high-k power ratio (final step): {summary['test_high_k_power_ratio_final_mean']:.4f}")
    print(f"  peak ratio ρ_max (mean):         {summary['test_mean_peak_ratio_rho_max']:.4f}  (<1 = underestimate)")
    print(f"  high-tail bias (mean):           {summary['test_mean_high_tail_bias']:.6f}  (<0 = underestimate highs)")
    print(f"  vorticity events truth/pred:     {summary['test_vort_event_count_truth']}/{summary['test_vort_event_count_pred']}")
    print(f">>> wrote {args.out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
