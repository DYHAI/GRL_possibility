#!/usr/bin/env bash
# Train + evaluate KH U-Net on 200-member Trixi ensemble (180 train / 20 test).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KH="$ROOT/experiments/kelvin_helmholtz"
DATA="${KH_DATA_DIR:-$ROOT/outputs/kelvin_helmholtz/trixi_ensemble_200}"
OUT="${KH_UNET_OUT:-$ROOT/outputs/kelvin_helmholtz/unet_mse}"
EPOCHS="${KH_EPOCHS:-30}"
BATCH="${KH_BATCH:-4}"

echo "=== data: $DATA ==="
echo "=== out:  $OUT ==="

python3 -u "$KH/train_unet.py" \
  --data-dir "$DATA" \
  --out-dir "$OUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH"

python3 -u "$KH/eval_unet.py" \
  --data-dir "$DATA" \
  --checkpoint "$OUT/unet_best.pt" \
  --split-json "$OUT/split.json" \
  --out-dir "$OUT/eval"

python3 -u "$KH/make_paper_figures.py" \
  --data-dir "$DATA" \
  --checkpoint "$OUT/unet_best.pt" \
  --split-json "$OUT/split.json" \
  --eval-json "$OUT/eval/eval_summary.json" \
  --out-dir "$ROOT/outputs/kelvin_helmholtz/figures"

echo "=== done: $OUT ==="
