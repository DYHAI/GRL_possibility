#!/usr/bin/env bash
# t=0→3, sampled frames, single-panel GIF (32+AMR, fast).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRIXI="$ROOT/experiments/kelvin_helmholtz/trixi"
OUT="$ROOT/outputs/kelvin_helmholtz/trixi_gif_3s"
JULIA="${JULIA:-$ROOT/tools/julia-1.11.9/bin/julia}"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"
export KH_REF=5
export KH_T_END=3.0
export KH_AMR=1
export KH_SAVE_INTERVAL="${KH_SAVE_INTERVAL:-40}"
export KH_OUT="$OUT"

echo "=== 1/3 Simulate t=0..3 (32+AMR, save every $KH_SAVE_INTERVAL steps) ==="
"$JULIA" "$TRIXI/run_kh_gif_3s.jl"

echo "=== 2/3 Export snapshots ==="
NVIS=8 "$JULIA" "$TRIXI/export_npz.jl" "$OUT"

echo "=== 3/3 Render GIF + HD PNG ==="
python3 "$ROOT/experiments/kelvin_helmholtz/render_trixi_gif.py" --data-dir "$OUT"

echo "Done: $OUT/kh_3s.gif"
