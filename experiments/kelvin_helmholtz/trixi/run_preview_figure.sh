#!/usr/bin/env bash
# 512×512 uniform KH, short time, ONE dual-panel figure.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRIXI="$ROOT/experiments/kelvin_helmholtz/trixi"
OUT="$ROOT/outputs/kelvin_helmholtz/trixi_512"
JULIA="${JULIA:-$ROOT/tools/julia-1.11.9/bin/julia}"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"
export KH_REF="${KH_REF:-9}"          # 2^9 = 512 cells / side
export KH_T_END="${KH_T_END:-0.5}"    # short preview
export KH_AMR="${KH_AMR:-0}"          # uniform 512×512
export KH_OUT="$OUT"

echo "=== 512×512 preview  t=0..$KH_T_END  (one figure only) ==="
echo "=== 1/3 Simulate ==="
"$JULIA" "$TRIXI/run_kh_preview.jl"

echo "=== 2/3 Export final frame ==="
NVIS=1 "$JULIA" "$TRIXI/export_npz.jl" "$OUT"

echo "=== 3/3 Render ONE PNG ==="
python3 "$ROOT/experiments/kelvin_helmholtz/render_trixi_hd.py" \
  --data-dir "$OUT" --grid-n 512 --dpi 200

echo ""
echo "Done: $OUT/hd/kh_hd_step*.png"
