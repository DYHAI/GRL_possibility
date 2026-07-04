# Agent guide — GRL_possibility

Context for Cursor / coding agents working in this repo.

## Paper goal

Demonstrate that **RMSE-optimal (MSE) deterministic forecasting** learns the **conditional mean**, which:

- over-smooths fine structure
- underestimates extremes
- can violate conservation / physical constraints

Two experiment tracks: **Markov matrices** (done) and **Kelvin–Helmholtz fluid** (active).

## Current focus: KH + Trixi → U-Net

Primary backend is **Julia/Trixi** for simulation; **PyTorch U-Net** for one-step forecasting.

| What | Path |
|------|------|
| Julia binary | `tools/julia-1.11.9/bin/julia` |
| Julia project | `experiments/kelvin_helmholtz/trixi/` |
| Shared setup | `experiments/kelvin_helmholtz/trixi/kh_common.jl` |
| 200-member ensemble | `experiments/kelvin_helmholtz/trixi/run_ensemble_200.sh` |
| Ensemble output | `outputs/kelvin_helmholtz/trixi_ensemble_200/member_*/member_*_512x4.npz` |
| Dataset loader | `experiments/kelvin_helmholtz/kh_dataset.py` |
| Train U-Net | `experiments/kelvin_helmholtz/train_unet.py` |
| Eval (rollout + spectra) | `experiments/kelvin_helmholtz/eval_unet.py` |
| Paper figures | `experiments/kelvin_helmholtz/make_paper_figures.py` |
| Full U-Net pipeline | `experiments/kelvin_helmholtz/run_unet_pipeline.sh` or `python run_kh_pipeline.py` |
| U-Net model | `experiments/common/models.py` (`UNet`, `radial_energy`) |
| Colormap | `experiments/kelvin_helmholtz/kh_colormap.py` |

### U-Net task (current spec)

- **Input / output:** previous frame → next frame, shape `(4, 512, 512)` for `(ρ, v₁, v₂, p)`
- **Loss:** global per-pixel RMSE on normalized fields
- **Split:** 200 members → **180 train / 20 test** (`split.json`, seed=42)
- **Eval metrics:**
  - one-step and long autoregressive rollout RMSE vs truth member
  - rollout vs **ensemble mean** (Monte Carlo conditional mean)
  - **radial power spectrum** of ρ, especially **high-k** tail during rollout
  - **extreme events** (`extreme_metrics.py`): peak amplitude ratio, tail exceedance, vorticity events — see README
- **Outputs:** `outputs/kelvin_helmholtz/unet_mse/` (checkpoints, `split.json`, `eval/`)

Default sim: **ref=5 + AMR**, stochastic per-step noise `step_rel_eps=3e-4`, `save_dt=0.2`, `t_end=5`.

## Visualization rules (important)

1. **Single panel only** — full domain `[-1,1]²`. No dual-panel zoom inset.
2. **Colormap:** `KH_CMAP`, `RHO_VMIN=0.45`, `RHO_VMAX=1.95` from `kh_colormap.py`.
3. **Orientation:** horizontal shear layers. `imshow(..., origin="lower")` **without** `.T` on griddata output.
4. **VTK coordinate swap:** cell center `cx = mean(pts[2])`, `cy = mean(pts[1])`.

## Markov experiment (done)

```bash
python experiments/markov/markov_mlp.py --sizes 3 10 100
```

Results: `outputs/markov/summary.json`

## Removed / out of scope

- **Cylinder wake experiment** — dropped; do not recreate unless user asks.

## Python KH fallback

`simulate.py`, `dgsem_euler.py`, `run_ensemble_parallel.py` — legacy; **not** used for U-Net training data. Use Trixi ensemble NPZ.

## Common pitfalls

- Ensemble members may **DtNaN before 5 s** — per-member NPZ keeps all survived frames; train/eval use whatever frames exist.
- `train_unet.py` needs **≥21 members** with `512x4.npz` before training (180/20 split).
- Re-export one member: `export_one_member_4var.jl` + `export_unet_grid.py --member-id N`.
- Only commit when user explicitly asks.

## Repo layout

```
GRL_possibility/
├── run_kh_pipeline.py         # train → eval → figures (Trixi NPZ)
├── experiments/
│   ├── common/models.py       # UNet, radial_energy
│   ├── markov/
│   └── kelvin_helmholtz/
│       ├── kh_dataset.py
│       ├── train_unet.py
│       ├── eval_unet.py
│       ├── make_paper_figures.py
│       ├── run_unet_pipeline.sh
│       └── trixi/             # Julia backend
├── tools/julia-1.11.9/
└── outputs/
    ├── markov/
    └── kelvin_helmholtz/
        ├── trixi_ensemble_200/
        ├── unet_mse/
        └── figures/
```
