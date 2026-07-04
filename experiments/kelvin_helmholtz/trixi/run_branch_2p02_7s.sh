#!/usr/bin/env bash
# Reference t=0→2.02 (no noise), then 5 members t=2.02→7 with per-step noise 3e-4.
# Usage:
#   bash run_branch_2p02_7s.sh
#   nohup bash run_branch_2p02_7s.sh &
#   tail -f outputs/kelvin_helmholtz/trixi_ensemble_branch_2p02_7s/run.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRIXI="$ROOT/experiments/kelvin_helmholtz/trixi"
OUT="$ROOT/outputs/kelvin_helmholtz/trixi_ensemble_branch_2p02_7s"
JULIA="${JULIA:-$ROOT/tools/julia-1.11.9/bin/julia}"
LOG="$OUT/run.log"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"
export KH_REF=5
export KH_AMR=1
export KH_T_BRANCH=2.02
export KH_T_END=7.0
export KH_IC_SEED=1
export KH_N_MEMBERS=5
export KH_SAVE_INTERVAL=40
export KH_SAVE_DT="${KH_SAVE_DT:-0.1}"
export KH_STEP_REL_EPS="${KH_STEP_REL_EPS:-3e-4}"
export KH_OUT="$OUT"

mkdir -p "$OUT"
{
echo "=== $(date -Iseconds) branch ensemble: 0→${KH_T_BRANCH}s ref + 5×${KH_T_BRANCH}→${KH_T_END}s step_eps=${KH_STEP_REL_EPS} ==="

echo "=== 1/3 Simulate (baseline + $KH_N_MEMBERS members) ==="
"$JULIA" "$TRIXI/run_kh_branch_2p02_7s.jl"

echo "=== 2/3 Export NPZ ==="
NVIS=8 "$JULIA" "$TRIXI/export_npz_ensemble.jl" "$OUT"

echo "=== 3/3 Render GIF ==="
python3 "$ROOT/experiments/kelvin_helmholtz/render_ensemble_gif.py" \
  --data-dir "$OUT" --grid-n 512 --duration-ms 120

if [[ "${KH_EXPORT_UNET:-0}" != "0" ]]; then
  echo "=== 4/4 Export U-Net 512×512×4 ==="
  python3 "$ROOT/experiments/kelvin_helmholtz/export_unet_grid.py" \
    --data-dir "$OUT" --grid-n "${KH_UNET_GRID:-512}" --nvis "${NVIS:-8}"
fi

echo "=== Done: $OUT/kh_ensemble_branch.gif ==="
} 2>&1 | tee "$LOG"
