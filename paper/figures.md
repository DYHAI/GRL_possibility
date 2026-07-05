# Figure checklist

## Markov experiment (Section 3–4)

| Fig | File (after generation) | Command |
|-----|-------------------------|---------|
| Transition matrix K=3 | `Figs/transition_matrices_1.png` | `python experiments/markov/markov_mlp.py --replot-only --sizes 3` |
| Transition matrix K=10 | `Figs/transition_matrices_2.png` | `--sizes 10` |
| Transition matrix K=100 | `Figs/transition_matrices_3.png` | `--sizes 100` |

Panel titles: **True P**, **MLP (RMSE)**, **MLP (Cross-Entropy)**. Axes: \(x_t\), \(x_{t+1}\).

Metrics table: RMSE + KL from `outputs/markov/summary.json`.

---

## KH / U-Net experiment (Section 3–4)

Generate after training + eval:

```bash
python run_kh_pipeline.py --epochs 30 --batch-size 2
# or train/eval/figures separately — see README.md
```

| Fig | Path | Content |
|-----|------|---------|
| Truth vs forecast ρ | `outputs/kelvin_helmholtz/figures/rollout_rho_member_XXX_step_YY.png` | Side-by-side density (yellow–blue) |
| High-k ratio (all test) | `outputs/kelvin_helmholtz/figures/high_k_ratio_all_test_members.png` | Over-smoothing |
| Per-member spectrum | `outputs/kelvin_helmholtz/unet_mse/eval/spectra/member_XXX_spectrum.png` | log-log ρ power + ratio vs step |
| Peak amplitude | `outputs/kelvin_helmholtz/unet_mse/eval/extremes/member_XXX_peaks.png` | truth / U-Net / ensemble max ρ |
| Tail exceedance | `outputs/kelvin_helmholtz/unet_mse/eval/extremes/exceedance_curve.png` | Pooled 20 test members |
| RMSE summary bar | `outputs/kelvin_helmholtz/unet_mse/eval/metrics_summary.png` | one-step + rollout metrics |

**Paper narrative:** U-Net curves should track **ensemble mean** more closely than **truth** on peaks, spectra, and tails.

---

## Optional (not auto-generated)

- Three-panel ρ: truth | U-Net | ensemble mean (same step) — add to `make_paper_figures.py` if needed.
- Markov scalar regression \(\mu(s)\) vs identity: `outputs/markov/nK/regression_curve.png` (from full train run).

---

## Colormap / layout rules

- Single full-domain panel \([-1,1]^2\); **no zoom inset**.
- `kh_colormap.py`: `KH_CMAP`, `RHO_VMIN=0.45`, `RHO_VMAX=1.95`.
- `imshow(..., origin="lower")` without `.T`.
