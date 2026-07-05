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
| Full U-Net pipeline | `python run_kh_pipeline.py` or `run_unet_pipeline.sh` |
| U-Net model | `experiments/common/models.py` (`UNet`, `radial_energy`) |
| Paper notes & LaTeX | `paper/MEMORY.md`, `paper/latex/` |

### U-Net task (current spec)

- **Input / output:** previous frame → next frame, shape `(4, 512, 512)` for `(ρ, v₁, v₂, p)`
- **Model:** U-Net, `base_ch=32`, **~1.93M parameters**
- **Loss:** global per-pixel RMSE on normalized fields (mean/std from **train only**)
- **Split (200 members, seed=42):** `split_members_three_way` → **170 train / 10 val / 20 test**
  - Val: early stopping in `train_unet.py` only
  - Test: `eval_unet.py` and paper figures — **never used in training**
  - Saved in `split.json` as `train_ids`, `val_ids`, `test_ids`
- **Train sampling:** `RandomOneStepDataset` — random member + consecutive `(t, t+1)` each step
- **Eval metrics:**
  - one-step and long autoregressive rollout RMSE vs truth member
  - rollout vs **ensemble mean** (Monte Carlo conditional mean)
  - **radial power spectrum** of ρ, especially **high-k** tail during rollout
  - **extreme events** (`extreme_metrics.py`): peak amplitude ratio, tail exceedance, vorticity events — see README
- **Outputs:** `outputs/kelvin_helmholtz/unet_mse/` (checkpoints, `split.json`, `eval/`)

Default sim: **ref=5 + AMR**, stochastic per-step noise `step_rel_eps=3e-4`, `save_dt=0.2`, `t_end=5`.

### Data & memory

| Item | Value |
|------|-------|
| 200-member disk total | ~19 GB (NPZ ~15 GB + H5 ~5 GB if `KH_KEEP_H5=1`) |
| Frames per member | typically 16–26 (may DtNaN before 5 s) |
| Train pairs | ~3600 |
| **Do not build unified `.pt`** on user's machine | OOM risk (~13 GB RAM to hold 170 NPZ + build overhead) |

Train from NPZ directly:

```bash
python3 experiments/kelvin_helmholtz/train_unet.py \
  --data-dir outputs/kelvin_helmholtz/trixi_ensemble_200 \
  --max-members 200 --n-val 10 --n-test 20 \
  --random-train --batch-size 2 --num-workers 0
```

Note: `train_unet.py` currently **loads all train members into RAM** at startup via `load_members()`. On 16 GB RAM this is tight (~13 GB). Prefer `batch-size 2`, `num-workers 0`. Lazy per-member loading is a future improvement if OOM occurs.

### Ensemble resume

`run_ensemble_200.sh` skips members that already have `member_XXX_512x4.npz`.

```bash
KH_START_MEMBER=186 bash experiments/kelvin_helmholtz/trixi/run_ensemble_200.sh
```

To re-run one member from scratch: delete its `member_XXX/` directory first.

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
- **Unified `.pt` dataset packing** — user opted out due to RAM; scripts exist (`build_unet_pt.py`, `pack_unet_dataset.py`) but are not the default workflow.

## Python KH fallback

`simulate.py`, `dgsem_euler.py`, `run_ensemble_parallel.py` — legacy; **not** used for U-Net training data. Use Trixi ensemble NPZ.

## Common pitfalls

- Ensemble members may **DtNaN before 5 s** — per-member NPZ keeps all survived frames; train/eval use whatever frames exist.
- `train_unet.py` needs **≥31 members** before training (170 + 10 + 20 split with at least 1 extra buffer).
- Do **not** confuse `val_ids` (10, tuning) with `test_ids` (20, final eval). Legacy `split.json` files may only have `test_ids` (= old val set).
- Re-export one member: `export_one_member_4var.jl` + `export_unet_grid.py --member-id N`.
- Only commit when user explicitly asks.
- GitHub remote: `https://github.com/DYHAI/GRL_possibility` — `outputs/`, `*.npz`, `*.h5`, `tools/` are gitignored.

## Paper manuscript

Consolidated notes for the AGU/GRL draft:

| Path | Content |
|------|---------|
| `paper/MEMORY.md` | **中文论文记忆**：论点、实验、指标、待办 |
| `paper/outline.md` | Section roadmap |
| `paper/figures.md` | Figure checklist + regenerate commands |
| `paper/latex/*.tex` | Copy-paste LaTeX blocks (theory, methods, discussion, data availability) |

Main thesis: MSE/RMSE-optimal deterministic forecast = conditional mean → over-smoothing, underestimated extremes, nonlinear non-closure (Jensen).

## Repo layout

```
GRL_possibility/
├── paper/                     # manuscript notes + LaTeX fragments
│   ├── MEMORY.md              # 论文记忆（中文）
│   └── latex/
├── run_kh_pipeline.py         # train → eval → figures (Trixi NPZ)
├── experiments/
│   ├── common/models.py       # UNet, radial_energy
│   ├── markov/
│   └── kelvin_helmholtz/
│       ├── kh_dataset.py      # split_members_three_way, datasets
│       ├── train_unet.py
│       ├── eval_unet.py
│       ├── make_paper_figures.py
│       ├── run_unet_pipeline.sh
│       └── trixi/             # Julia backend
├── tools/julia-1.11.9/        # gitignored
└── outputs/                   # gitignored
    ├── markov/
    └── kelvin_helmholtz/
        ├── trixi_ensemble_200/
        ├── unet_mse/
        └── figures/
```
