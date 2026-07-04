#!/usr/bin/env bash
# One trajectory → one dual-panel PNG (fast preview pipeline).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JULIA="${ROOT}/tools/julia-1.11.9/bin/julia"
PREVIEW="${ROOT}/outputs/kelvin_helmholtz/trixi_preview"
KH="${ROOT}/experiments/kelvin_helmholtz"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"
export NVIS="${NVIS:-4}"

echo "=== 1/3 Simulate (single traj, short t) ==="
cd "${ROOT}/experiments/kelvin_helmholtz/trixi"
"${JULIA}" run_kh_preview.jl

echo "=== 2/3 Export final frame to NPZ ==="
rm -rf "${PREVIEW}/vtu_converted"
"${JULIA}" export_npz.jl "${PREVIEW}"

echo "=== 3/3 Render ONE HD figure ==="
cd "${KH}"
python3 render_trixi_hd.py \
  --data-dir "${PREVIEW}" \
  --step "$(python3 -c "import numpy as np; s=sorted(int(x) for x in np.load('${PREVIEW}/snapshots.npz')['steps']); print(s[-1])")" \
  --grid-n 1024 \
  --dpi 200

echo ""
echo "Done. Figure:"
ls -la "${PREVIEW}/hd/"*.png 2>/dev/null || true
