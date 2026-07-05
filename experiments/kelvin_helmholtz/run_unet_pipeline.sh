#!/usr/bin/env bash
# Train U-Net from per-member NPZ (no .pt packing — saves RAM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KH="$ROOT/experiments/kelvin_helmholtz"
DATA="${KH_DATA_DIR:-$ROOT/outputs/kelvin_helmholtz/trixi_ensemble_200}"
OUT="${KH_UNET_OUT:-$ROOT/outputs/kelvin_helmholtz/unet_mse}"
EPOCHS="${KH_EPOCHS:-30}"
BATCH="${KH_BATCH:-2}"

echo "=== Train U-Net from NPZ (170/10/20 split) ==="
python3 -u "$KH/train_unet.py" \
  --data-dir "$DATA" \
  --out-dir "$OUT" \
  --max-members 200 \
  --n-val 10 \
  --n-test 20 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --num-workers 0 \
  --random-train

echo "=== Eval on 20 test members ==="
python3 -u "$KH/eval_unet.py" \
  --data-dir "$DATA" \
  --checkpoint "$OUT/unet_best.pt" \
  --split-json "$OUT/split.json" \
  --out-dir "$OUT/eval"

echo "=== Paper figures ==="
python3 -u "$KH/make_paper_figures.py" \
  --data-dir "$DATA" \
  --checkpoint "$OUT/unet_best.pt" \
  --split-json "$OUT/split.json" \
  --eval-json "$OUT/eval/eval_summary.json" \
  --out-dir "$ROOT/outputs/kelvin_helmholtz/figures"

echo "=== done: $OUT ==="
