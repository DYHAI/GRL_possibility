#!/usr/bin/env bash
# FAST preview: Trixi elixir mesh (32 + AMR), ~15–30 s sim + one PNG.
# Much faster than uniform 512² or 1024².
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRIXI="$ROOT/experiments/kelvin_helmholtz/trixi"
OUT="$ROOT/outputs/kelvin_helmholtz/trixi_fast"
JULIA="${JULIA:-$ROOT/tools/julia-1.11.9/bin/julia}"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"
export KH_REF="${KH_REF:-5}"           # elixir: 32 base cells
export KH_T_END="${KH_T_END:-1.0}"     # enough to see roll-up
export KH_AMR="${KH_AMR:-1}"           # AMR on → refine only at interface
export KH_OUT="$OUT"

echo "=== FAST preview (32+AMR)  t=0..$KH_T_END  → one PNG ==="
"$JULIA" "$TRIXI/run_kh_preview.jl"

echo "=== Export + render ==="
NVIS=8 "$JULIA" "$TRIXI/export_npz.jl" "$OUT"
python3 "$ROOT/experiments/kelvin_helmholtz/render_trixi_hd.py" \
  --data-dir "$OUT" --grid-n 1024 --dpi 200

echo "Done: $OUT/hd/kh_hd_step*.png"
