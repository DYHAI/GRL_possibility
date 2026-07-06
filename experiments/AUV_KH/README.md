# AUV_KH — Navigate through Kelvin–Helmholtz flow fields

Toy RL playground: an **AUV** (autonomous underwater vehicle) must go from **top-left** to **bottom-right** on the KH domain `[-1,1]²`, affected by:

- **control thrust** (2D action)
- **local flow velocity** `(v₁, v₂)` from ensemble NPZ members
- **linear drag**

## Data split (200 members, seed=42)

| Set | Count | Use |
|-----|-------|-----|
| Train | **180** | RL training playground (random member each episode) |
| Test | **20** | Generalization eval only |

Same shuffle logic as `split_members_three_way(..., n_val=0, n_test=20)`.

## Physics (simplified)

```
v ← (1 - drag)·v + thrust_scale·action·dt + flow_scale·u_flow·dt
x ← x + v·dt
```

Flow is bilinearly sampled from downsampled `v1,v2` grids (default 64×64) at the current member frame (time advances each step).

## Install

```bash
pip install gymnasium stable-baselines3
```

## Train

```bash
# PPO (default 200k steps)
python3 experiments/AUV_KH/train_rl.py \
  --data-dir /home/ding/桌面/Interests/200_ensemble \
  --algo ppo

# TD3
python3 experiments/AUV_KH/train_rl.py --algo td3

# Both + eval
bash experiments/AUV_KH/run_pipeline.sh
```

## Eval (20 test members)

```bash
python3 experiments/AUV_KH/eval_rl.py \
  --checkpoint outputs/auv_kh/rl/ppo_auv_kh \
  --split-json outputs/auv_kh/rl/split.json \
  --algo ppo
```

Outputs: `outputs/auv_kh/rl/split.json`, `ppo_auv_kh.zip`, `td3_auv_kh.zip`, `eval_*.json`.

## Files

| File | Role |
|------|------|
| `kh_flow_cache.py` | Lazy NPZ velocity loader |
| `auv_kh_env.py` | Gymnasium environment |
| `train_rl.py` | PPO / TD3 training |
| `eval_rl.py` | Test-member generalization |
