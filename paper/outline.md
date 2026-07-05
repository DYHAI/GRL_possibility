# Section outline

## Introduction
- AI weather/ocean models (Pangu, FuXi, GraphCast) optimize MSE/RMSE.
- Roadmap → `latex/intro_roadmap.tex`

## Section 2 — Theory (`sec:preliminaries` / `sec:theory`)
1. Markov formulation of forecasting → `sec02_markov_formulation.tex`
2. MSE training → population risk → **Proposition**: MSE-optimal = conditional mean → `sec02_mse_conditional_mean.tex`
3. **Corollary**: weighted average (integral / sum over states)
4. Three mechanisms (over-smoothing, extremes, non-conservation) — preview or short remarks
5. **Proposition**: conditional mean not a nonlinear solution + Jensen → `sec02_jensen_nonclosure.tex`

## Section 3 — Data and Methods (`sec:methods`)
1. Markov toy experiment → `sec03_markov_experiment.tex`
2. KH Trixi ensemble → `sec03_kh_trixi.tex`
3. U-Net one-step forecaster → `sec03_unet.tex`
4. Computational environment → `computational_environment.tex`

## Section 4 — Results and Discussion (`sec:results_discussion`)
1. Brief experimental recap (Markov + KH)
2. Three mechanisms in depth → `sec04_discussion_three_mechanisms.tex`
3. Link metrics to theory (placeholders for numbers)

## Back matter
- Data Availability → `data_availability.tex`
- Conclusion (user draft)

## Key labels
- `prop:mse_conditional_mean`
- `cor:weighted_average`
- `prop:nonlinear_nonclosure`
- `subsec:markov_methods`
- `subsec:kh_methods`
