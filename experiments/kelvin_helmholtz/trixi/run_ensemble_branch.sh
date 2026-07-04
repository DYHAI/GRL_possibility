#!/usr/bin/env bash
# Stochastic ensemble: t=0→5 s, 5 members, per-step noise, per-member 512×512×4 NPZ.
# Usage:
#   bash run_ensemble_branch.sh
#   nohup bash run_ensemble_branch.sh &
#   tail -f outputs/kelvin_helmholtz/trixi_ensemble_branch/run.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRIXI="$ROOT/experiments/kelvin_helmholtz/trixi"
OUT="$ROOT/outputs/kelvin_helmholtz/trixi_ensemble_branch"
JULIA="${JULIA:-$ROOT/tools/julia-1.11.9/bin/julia}"
LOG="$OUT/run.log"
NVIS="${NVIS:-8}"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"
export KH_REF=5
export KH_AMR=1
export KH_T_END=5.0
export KH_IC_SEED=1
export KH_N_MEMBERS=5
export KH_SAVE_INTERVAL=40
export KH_SAVE_DT="${KH_SAVE_DT:-0.2}"
export KH_STEP_REL_EPS="${KH_STEP_REL_EPS:-3e-4}"
export KH_EXPORT_ALIGN="${KH_EXPORT_ALIGN:-0}"
export KH_OUT="$OUT"

mkdir -p "$OUT"
{
echo "=== $(date -Iseconds) stochastic ensemble: 5 members t=0→${KH_T_END}s, save_dt=${KH_SAVE_DT}s, step_eps=${KH_STEP_REL_EPS}, export_align=${KH_EXPORT_ALIGN} ==="

echo "=== 1/4 Simulate ($KH_N_MEMBERS members) ==="
"$JULIA" "$TRIXI/run_kh_ensemble_branch.jl"

echo "=== 2/4 Export cell-center 4-var NPZ (per member) ==="
NVIS="$NVIS" "$JULIA" "$TRIXI/export_ensemble_4var.jl" "$OUT"

echo "=== 3/4 Export 512×512×4 grids (one NPZ per member) ==="
python3 "$ROOT/experiments/kelvin_helmholtz/export_unet_grid.py" \
  --data-dir "$OUT" --grid-n "${KH_UNET_GRID:-512}" --skip-julia

echo "=== 4/4 Render GIF (common times only) ==="
python3 "$ROOT/experiments/kelvin_helmholtz/render_ensemble_gif.py" \
  --data-dir "$OUT" --grid-n 512 --duration-ms 120

echo "=== Done: $OUT/member_*/member_*_512x4.npz ==="
} 2>&1 | tee "$LOG"
