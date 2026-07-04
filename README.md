# GRL_possibility Experiments

Controlled experiments for the paper on **RMSE-optimal deterministic forecasting** = conditional mean → over-smoothing, underestimated extremes, conservation issues.

## Experiments

| # | Experiment | Status | Backend |
|---|------------|--------|---------|
| 1 | Markov transition matrix (3 / 10 / 100 states) | **Done** | Python |
| 2 | Kelvin–Helmholtz instability + U-Net MSE | **In progress** | Julia/Trixi (primary), Python (fallback) |

---

## 1. Markov (done)

Learn row-stochastic transition matrices with:

- **MLP + RMSE** → scalar regression, approximates conditional mean
- **MLP + cross-entropy** → full transition row

```bash
python experiments/markov/markov_mlp.py --sizes 3 10 100
```

Outputs: `outputs/markov/n{3,10,100}/`, summary in `outputs/markov/summary.json`

---

## 2. Kelvin–Helmholtz (KH)

2D compressible Euler KH instability. Stochastic IC (random phase/amplitude per seed) gives an ensemble of trajectories. Goal: train U-Net with MSE on next-step density, show smoothed / under-resolved forecasts vs truth.

**Production solver:** Trixi DGSEM (p=3, Ranocha + shock capturing HG, Lax–Friedrichs, AMR). Matches elixir `elixir_euler_kelvin_helmholtz_instability_amr.jl`.

**Visualization:** yellow–blue density colormap, **single full-domain panel only** (no zoom inset). See `experiments/kelvin_helmholtz/kh_colormap.py`.

### U-Net one-step forecaster (512×512×4)

Train on Trixi ensemble NPZ: predict next frame from previous frame for all four fields `(ρ, v₁, v₂, p)`.

| Setting | Value |
|---------|-------|
| Input / output | `(4, 512, 512)` |
| Loss | per-pixel RMSE |
| Train / test | 180 / 20 members (seed=42) |
| Eval | long rollout, ρ power spectrum (high-k), vs truth & ensemble mean |

### Extreme-event comparison

RMSE-optimal forecasts should **underestimate peaks** and **narrow tails**. Compare three fields on test rollout:

| Reference | Role |
|-----------|------|
| **Truth member** | single stochastic trajectory |
| **U-Net rollout** | deterministic RMSE-optimal forecast |
| **Ensemble mean** | Monte Carlo estimate of conditional mean E[x\|x₀] |

**Thresholds** (Q01/Q95/Q99 of ρ, Q95 of max vorticity) are computed from **train members only** — no test leakage.

Metrics (auto in `eval_unet.py` → `eval/extremes/`):

1. **Peak amplitude** — domain max ρ vs rollout step; ratio forecast/truth (< 1 ⇒ underestimation)
2. **Tail exceedance** — P(ρ ≥ threshold) curve pooled over test rollout
3. **Conditional tail bias** — mean(forecast − truth) on pixels where truth ρ is in top 5% (train-defined)
4. **Vorticity events** — count frames with max|ω| above train Q95; compare truth vs forecast event frequency & intensity
5. **Gradient peaks** — max |∇ρ| ratio (interface sharpness)

Expected paper narrative: forecast peak ratio ≈ ensemble mean < 1; high-tail bias < 0; fewer/weaker vorticity events in rollout.

**Prerequisite:** 200-member ensemble with per-member NPZ:

```bash
# still running / resume-safe
bash experiments/kelvin_helmholtz/trixi/run_ensemble_200.sh
tail -f outputs/kelvin_helmholtz/trixi_ensemble_200/run.log
```

**Train + eval + figures:**

```bash
bash experiments/kelvin_helmholtz/run_unet_pipeline.sh
# or
python run_kh_pipeline.py --epochs 30
```

Outputs:

- `outputs/kelvin_helmholtz/unet_mse/unet_best.pt` — checkpoint + norm stats
- `outputs/kelvin_helmholtz/unet_mse/split.json` — train/test member IDs
- `outputs/kelvin_helmholtz/unet_mse/eval/eval_summary.json` — rollout metrics & spectra
- `outputs/kelvin_helmholtz/figures/` — ρ snapshots, high-k ratio plots

Individual steps:

```bash
python3 experiments/kelvin_helmholtz/train_unet.py \
  --data-dir outputs/kelvin_helmholtz/trixi_ensemble_200

python3 experiments/kelvin_helmholtz/eval_unet.py \
  --checkpoint outputs/kelvin_helmholtz/unet_mse/unet_best.pt

python3 experiments/kelvin_helmholtz/make_paper_figures.py
```

### 200-member ensemble (in progress)

Stochastic IC + per-step noise (`step_rel_eps=3e-4`), `save_dt=0.2 s`, `t=0→5 s`:

```bash
bash experiments/kelvin_helmholtz/trixi/run_ensemble_200.sh
```

Per member: `member_XXX/member_XXX_512x4.npz` shape `(T, 4, 512, 512)`. H5 kept by default (~+5 GB total vs NPZ-only ~15 GB).

Key env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `KH_N_MEMBERS` | 200 | Ensemble size |
| `KH_SAVE_DT` | 0.2 | Snapshot interval (s) |
| `KH_STEP_REL_EPS` | 3e-4 | Per-step relative noise |
| `KH_KEEP_H5` | 1 | Keep raw H5 after export |
| `KH_CLEAN` | 0 | Wipe output dir before run |

### Quick commands (visualization)

```bash
# Fast preview (~30 s): sim t=0→1, one HD PNG
bash experiments/kelvin_helmholtz/trixi/run_preview_fast.sh

# Full GIF t=0→3 (32+AMR, ~28 s sim + render)
bash experiments/kelvin_helmholtz/trixi/run_gif_3s.sh

# Re-render GIF only (snapshots.npz already exists)
python3 experiments/kelvin_helmholtz/render_trixi_gif.py \
  --data-dir outputs/kelvin_helmholtz/trixi_gif_3s
```

**Latest GIF output:** `outputs/kelvin_helmholtz/trixi_gif_3s/kh_3s.gif`

### Environment

- **Julia 1.11.9:** `tools/julia-1.11.9/bin/julia` (or set `JULIA=…`)
- **Project:** `experiments/kelvin_helmholtz/trixi/` (`Project.toml` / `Manifest.toml`)
- **Trixi source:** local copy at `experiments/Trixi.jl-main/`

Key env vars for Trixi runs:

| Variable | Default | Meaning |
|----------|---------|---------|
| `KH_REF` | 5 | Base mesh refinement (32 cells/side) |
| `KH_AMR` | 1 | AMR on/off |
| `KH_T_END` | 3.0 | Simulation end time |
| `KH_SAVE_INTERVAL` | 40 | Steps between snapshots |
| `KH_OUT` | (script-specific) | Output directory |
| `NVIS` | 8 | VTK visualization nodes (export step) |

### Pipeline layout

```
experiments/kelvin_helmholtz/
├── trixi/
│   ├── kh_common.jl          # shared IC, DGSEM+AMR setup
│   ├── run_kh_gif_3s.jl      # t=0→3 multi-frame sim
│   ├── export_npz.jl         # H5 → VTU → snapshots.npz
│   ├── run_gif_3s.sh         # sim → export → render (one command)
│   └── run_preview_fast.sh   # fast single-frame preview
├── render_trixi_gif.py       # GIF + HD still from snapshots.npz
├── render_trixi_hd.py        # standalone HD PNG
└── kh_colormap.py            # yellow–blue ρ colormap
```

### Python fallback (not production-ready)

`simulate.py`, `dgsem_euler.py`, `euler_fv.py` — attempted pure-Python DGSEM; unstable without full Trixi shock capturing. Use Trixi for paper figures.

### Still TODO (KH)

- [x] 200-member Trixi ensemble pipeline (`run_ensemble_200.sh`)
- [x] `train_unet.py`, `eval_unet.py`, `make_paper_figures.py`
- [ ] Finish 200-member data generation (wait for `trixi_ensemble_200/`)
- [ ] Full U-Net training run on 180/20 split
- [ ] Paper figures from final eval

---

## Run Markov only

```bash
python run_all.py
```

---

## Dependencies

**Python:**

```bash
pip install -r requirements.txt
```

**Julia** (for KH): activate `experiments/kelvin_helmholtz/trixi/` — deps include Trixi, Trixi2Vtk, ReadVTK, NPZ, OrdinaryDiffEqLowStorageRK.

---

## Agent notes

See [`AGENTS.md`](AGENTS.md) for Cursor/agent context (paths, pitfalls, current focus).
