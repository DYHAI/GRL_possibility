#!/usr/bin/env bash
# AUV-KH: PPO + TD3 on 180 train members, eval on 20 test members.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AUV="$ROOT/experiments/AUV_KH"
DATA="${AUV_DATA_DIR:-/home/ding/桌面/Interests/200_ensemble}"
OUT="${AUV_OUT:-$ROOT/outputs/auv_kh/rl}"
STEPS="${AUV_TIMESTEPS:-200000}"

echo "=== PPO ==="
python3 -u "$AUV/train_rl.py" --data-dir "$DATA" --out-dir "$OUT" --algo ppo --timesteps "$STEPS"

echo "=== TD3 ==="
python3 -u "$AUV/train_rl.py" --data-dir "$DATA" --out-dir "$OUT" --algo td3 --timesteps "$STEPS"

echo "=== Eval PPO ==="
python3 -u "$AUV/eval_rl.py" --checkpoint "$OUT/ppo_auv_kh" --split-json "$OUT/split.json" --algo ppo

echo "=== Eval TD3 ==="
python3 -u "$AUV/eval_rl.py" --checkpoint "$OUT/td3_auv_kh" --split-json "$OUT/split.json" --algo td3

echo "=== done: $OUT ==="
