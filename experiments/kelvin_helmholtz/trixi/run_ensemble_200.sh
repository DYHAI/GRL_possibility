#!/usr/bin/env bash
# 200-member incremental ensemble: sim -> 4var -> 512x4 per member, resume-safe.
#
# Usage:
#   bash run_ensemble_200.sh
#   KH_KEEP_H5=1 nohup bash run_ensemble_200.sh &
#   tail -f outputs/kelvin_helmholtz/trixi_ensemble_200/run.log
#
# Env:
#   KH_N_MEMBERS=200   KH_STEP_REL_EPS=3e-4   KH_SAVE_DT=0.2
#   KH_KEEP_H5=1       keep solution H5 in member_*/vtu/ (default on)
#   KH_DELETE_VTU=1    remove vtu_converted after export (default on, saves ~90MB/member)
#   KH_START_MEMBER=186  start from member N (default 1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRIXI="$ROOT/experiments/kelvin_helmholtz/trixi"
OUT="$ROOT/outputs/kelvin_helmholtz/trixi_ensemble_200"
JULIA="${JULIA:-$ROOT/tools/julia-1.11.9/bin/julia}"
LOG="$OUT/run.log"
NVIS="${NVIS:-8}"
GRID_N="${KH_UNET_GRID:-512}"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"
export KH_REF=5
export KH_AMR=1
export KH_T_END=5.0
export KH_IC_SEED=1
export KH_N_MEMBERS="${KH_N_MEMBERS:-200}"
KH_START_MEMBER="${KH_START_MEMBER:-1}"
export KH_SAVE_INTERVAL=40
export KH_SAVE_DT="${KH_SAVE_DT:-0.2}"
export KH_STEP_REL_EPS="${KH_STEP_REL_EPS:-3e-4}"
export KH_EXPORT_ALIGN=0
export KH_OUT="$OUT"

KEEP_H5="${KH_KEEP_H5:-1}"
DELETE_VTU="${KH_DELETE_VTU:-1}"

mkdir -p "$OUT"

if [[ "${KH_CLEAN:-0}" == "1" ]]; then
  echo "=== KH_CLEAN=1: wiping $OUT ==="
  rm -rf "$OUT"/*
  mkdir -p "$OUT"
fi

{
echo "=== $(date -Iseconds) ensemble ${KH_N_MEMBERS} members (start=${KH_START_MEMBER}) t=0→${KH_T_END}s save_dt=${KH_SAVE_DT} eps=${KH_STEP_REL_EPS} keep_h5=${KEEP_H5} ==="

done=0
skipped=0
for m in $(seq "$KH_START_MEMBER" "$KH_N_MEMBERS"); do
  mm=$(printf '%03d' "$m")
  member_dir="$OUT/member_${mm}"
  out_npz="$member_dir/member_${mm}_${GRID_N}x4.npz"

  if [[ -f "$out_npz" ]]; then
    skipped=$((skipped + 1))
    echo "--- member $m/$KH_N_MEMBERS skip (exists) ---"
    continue
  fi

  echo ""
  echo "=== member $m/$KH_N_MEMBERS $(date -Iseconds) ==="

  export KH_MEMBER_ID="$m"
  echo "  [1/3] simulate"
  "$JULIA" "$TRIXI/run_one_kh_member.jl"

  echo "  [2/3] export 4-var fields"
  NVIS="$NVIS" "$JULIA" "$TRIXI/export_one_member_4var.jl" "$OUT" "$m"

  echo "  [3/3] export ${GRID_N}x4 grid"
  python3 -u "$ROOT/experiments/kelvin_helmholtz/export_unet_grid.py" \
    --data-dir "$OUT" --grid-n "$GRID_N" --member-id "$m"

  if [[ "$DELETE_VTU" == "1" ]]; then
    rm -rf "$member_dir/vtu_converted"
    rm -f "$member_dir/member_${mm}_fields_4var.npz"
  fi
  if [[ "$KEEP_H5" != "1" ]]; then
    rm -rf "$member_dir/vtu"
  fi

  done=$((done + 1))
  echo "  member $m finished  done=$done skipped=$skipped"
done

echo ""
echo "=== $(date -Iseconds) complete: $OUT  processed=$done skipped=$skipped ==="
echo "=== outputs: member_*/member_*_${GRID_N}x4.npz ==="
} 2>&1 | tee -a "$LOG"
