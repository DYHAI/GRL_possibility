# GRL_possibility Experiments

Controlled experiments for the paper on **RMSE-optimal deterministic forecasting** = conditional mean → over-smoothing, underestimated extremes, conservation issues.

**Repo:** https://github.com/DYHAI/GRL_possibility

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

2D compressible Euler KH instability. Stochastic IC (random phase/amplitude per seed) gives an ensemble of trajectories. Goal: train U-Net with MSE on next-step fields, show smoothed / under-resolved forecasts vs truth.

**Production solver:** Trixi DGSEM (p=3, Ranocha + shock capturing HG, Lax–Friedrichs, AMR). Matches elixir `elixir_euler_kelvin_helmholtz_instability_amr.jl`.

**Visualization:** yellow–blue density colormap, **single full-domain panel only** (no zoom inset). See `experiments/kelvin_helmholtz/kh_colormap.py`.

### U-Net one-step forecaster (512×512×4)

Train on Trixi ensemble NPZ: predict next frame from previous frame for all four fields `(ρ, v₁, v₂, p)`.

| Setting | Value |
|---------|-------|
| Input / output | `(4, 512, 512)` |
| Model | U-Net, base=32, **~1.93M params** |
| Loss | global per-pixel RMSE |
| Split (200 members, seed=42) | **170 train / 10 val / 20 test** |
| Train sampling | random `(member, t)→(t+1)` pair each step |
| Val | 10 members — early stopping only |
| Test | 20 members — final eval only (`eval_unet.py`) |

**Data scale (200 members, full):**

| Item | Size |
|------|------|
| Whole ensemble dir (NPZ + H5) | **~19 GB** |
| NPZ only (`512×4`) | **~15 GB** |
| Per member | ~70–110 MB, ~16–26 frames (`save_dt=0.2 s`, may stop before 5 s) |
| Train pairs | **~3600** consecutive-frame pairs |

**Training time (RTX 4060 Laptop, 30 epochs):** roughly **4–10 h** with `batch-size 2–4`. On 16 GB RAM, use `batch-size 2 --num-workers 0` — loading all 170 train NPZ at once uses ~13 GB RAM.

**Do not pack into a single `.pt`** on this machine — it doubles memory use during build and training.

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

### 200-member ensemble

Stochastic IC + per-step noise (`step_rel_eps=3e-4`), `save_dt=0.2 s`, `t=0→5 s`:

```bash
# Full run (resume-safe: skips members with existing 512x4.npz)
bash experiments/kelvin_helmholtz/trixi/run_ensemble_200.sh
tail -f outputs/kelvin_helmholtz/trixi_ensemble_200/run.log

# Resume from member N (e.g. after crash)
KH_START_MEMBER=186 bash experiments/kelvin_helmholtz/trixi/run_ensemble_200.sh
```

Per member: `member_XXX/member_XXX_512x4.npz` shape `(T, 4, 512, 512)`.

Key env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `KH_N_MEMBERS` | 200 | Ensemble size |
| `KH_START_MEMBER` | 1 | Start loop at member N (resume) |
| `KH_SAVE_DT` | 0.2 | Snapshot interval (s) |
| `KH_STEP_REL_EPS` | 3e-4 | Per-step relative noise |
| `KH_KEEP_H5` | 1 | Keep raw H5 after export (~25 MB/member) |
| `KH_DELETE_VTU` | 1 | Remove intermediate VTU after export |
| `KH_CLEAN` | 0 | Wipe output dir before run |

### Train + eval + figures

```bash
# Recommended: NPZ → train → eval → figures
python run_kh_pipeline.py --epochs 30 --batch-size 2

# Or shell wrapper (same defaults)
bash experiments/kelvin_helmholtz/run_unet_pipeline.sh
```

Individual steps:

```bash
python3 experiments/kelvin_helmholtz/train_unet.py \
  --data-dir outputs/kelvin_helmholtz/trixi_ensemble_200 \
  --max-members 200 --n-val 10 --n-test 20 \
  --random-train --batch-size 2 --num-workers 0 --epochs 30

python3 experiments/kelvin_helmholtz/eval_unet.py \
  --data-dir outputs/kelvin_helmholtz/trixi_ensemble_200 \
  --checkpoint outputs/kelvin_helmholtz/unet_mse/unet_best.pt \
  --split-json outputs/kelvin_helmholtz/unet_mse/split.json

python3 experiments/kelvin_helmholtz/make_paper_figures.py \
  --split-json outputs/kelvin_helmholtz/unet_mse/split.json \
  --eval-json outputs/kelvin_helmholtz/unet_mse/eval/eval_summary.json
```

Outputs:

- `outputs/kelvin_helmholtz/unet_mse/unet_best.pt` — checkpoint + norm stats
- `outputs/kelvin_helmholtz/unet_mse/split.json` — `train_ids`, `val_ids`, `test_ids`
- `outputs/kelvin_helmholtz/unet_mse/eval/eval_summary.json` — rollout metrics & spectra
- `outputs/kelvin_helmholtz/figures/` — ρ snapshots, high-k ratio plots

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

- **Python:** 3.13+, PyTorch 2.12+ (CUDA 13.0 tested on RTX 4060 Laptop 8 GB)
- **Julia 1.11.9:** `tools/julia-1.11.9/bin/julia` (or set `JULIA=…`)
- **Project:** `experiments/kelvin_helmholtz/trixi/` (`Project.toml` / `Manifest.toml`)
- **Trixi source:** local copy at `experiments/Trixi.jl-main/`

Key env vars for Trixi runs:

| Variable | Default | Meaning |
|----------|---------|---------|
| `KH_REF` | 5 | Base mesh refinement (32 cells/side) |
| `KH_AMR` | 1 | AMR on/off |
| `KH_T_END` | 5.0 | Simulation end time (ensemble) |
| `KH_SAVE_INTERVAL` | 40 | Steps between snapshots |
| `KH_OUT` | (script-specific) | Output directory |
| `NVIS` | 8 | VTK visualization nodes (export step) |

### Pipeline layout

```
experiments/kelvin_helmholtz/
├── kh_dataset.py             # NPZ loader, 170/10/20 split
├── train_unet.py             # train (val for early stop)
├── eval_unet.py              # eval on 20 test members only
├── make_paper_figures.py
├── run_unet_pipeline.sh
├── trixi/
│   ├── kh_common.jl          # shared IC, DGSEM+AMR setup
│   ├── run_ensemble_200.sh   # 200-member incremental pipeline
│   ├── run_one_kh_member.jl
│   ├── export_one_member_4var.jl
│   └── export_unet_grid.py   # (in parent dir)
├── export_unet_grid.py       # NPZ 512×4 grid export
└── kh_colormap.py
```

### Python fallback (not production-ready)

`simulate.py`, `dgsem_euler.py`, `euler_fv.py` — attempted pure-Python DGSEM; unstable without full Trixi shock capturing. Use Trixi for paper figures.

### Still TODO (KH)

- [x] 200-member Trixi ensemble pipeline (`run_ensemble_200.sh`, resume via `KH_START_MEMBER`)
- [x] `train_unet.py`, `eval_unet.py`, `make_paper_figures.py`
- [x] Three-way split 170 / 10 / 20 (`split_members_three_way`)
- [ ] Finish 200-member data generation
- [ ] Full U-Net training run
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
