# Paper materials — GRL_possibility

Working notes and LaTeX fragments for the AGU/GRL manuscript on **RMSE-optimal forecasting ≈ conditional mean**.

| File | Purpose |
|------|---------|
| [`MEMORY.md`](MEMORY.md) | **论文记忆**（中文）：论点、实验设计、图表清单、写作状态 |
| [`outline.md`](outline.md) | Section-by-section roadmap (EN) |
| [`figures.md`](figures.md) | Required figures + repo paths to regenerate |
| [`latex/`](latex/) | Copy-paste LaTeX blocks per section |

## How to use

1. Read `MEMORY.md` first for the full narrative.
2. Paste blocks from `latex/*.tex` into your AGU template (`agujournal2019`).
3. Regenerate figures from this repo (see `figures.md`); large data stay in `outputs/` (not on GitHub).

## Main thesis (one paragraph)

Deterministic AI weather/ocean models trained with MSE/RMSE learn the **conditional mean** \(\mathbb{E}[\mathbf{X}_{t+\Delta t}\mid \mathbf{X}_t]\), not the full transition law \(p(\mathbf{x}_{t+\Delta t}\mid \mathbf{x}_t)\). This mean field (i) **low-pass filters** high-frequency variability, (ii) **underestimates extremes**, and (iii) is **not a solution** of the original nonlinear governing equations (Jensen / closure error).

## Experiments

| # | Experiment | Code | Status |
|---|------------|------|--------|
| 1 | Markov matrix K=3,10,100; MLP(RMSE) vs MLP(CE) | `experiments/markov/markov_mlp.py` | Done |
| 2 | KH compressible Euler, Trixi DGSEM+AMR, 200-member ensemble, U-Net | `experiments/kelvin_helmholtz/` | Data ~200 members; train/eval pending |

## Split (KH U-Net)

200 members → **170 train / 10 val / 20 test** (`seed=42`, `split_members_three_way` in `kh_dataset.py`).
