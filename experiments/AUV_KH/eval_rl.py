#!/usr/bin/env python3
"""Evaluate trained PPO/TD3 on held-out KH members (generalization)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.AUV_KH.auv_kh_env import AUVConfig, AUVKHNavEnv
from experiments.AUV_KH.kh_flow_cache import KHFlowCache, default_data_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=ROOT / "outputs/auv_kh/rl/split.json")
    parser.add_argument("--algo", choices=("ppo", "td3"), default="ppo")
    parser.add_argument("--episodes-per-member", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO, TD3
    except ImportError as e:
        raise SystemExit("pip install stable-baselines3") from e

    split = json.loads(args.split_json.read_text())
    data_dir = Path(split["data_dir"])
    test_ids = split["test_ids"]
    train_ids = split["train_ids"]

    Algo = PPO if args.algo == "ppo" else TD3
    model = Algo.load(str(args.checkpoint))

    cache = KHFlowCache(data_dir, test_ids, grid_n=64)
    rng = np.random.default_rng(args.seed)

    results = []
    for mid in test_ids:
        env = AUVKHNavEnv(cache, [mid], config=AUVConfig(), seed=int(rng.integers(0, 1_000_000)))
        succ, dists, rews = [], [], []
        for ep in range(args.episodes_per_member):
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            done = False
            total_r = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                total_r += r
                done = term or trunc
            succ.append(bool(info.get("success", False)))
            dists.append(float(info.get("dist", np.nan)))
            rews.append(total_r)
        results.append(
            {
                "member_id": mid,
                "success_rate": float(np.mean(succ)),
                "mean_final_dist": float(np.mean(dists)),
                "mean_return": float(np.mean(rews)),
            }
        )
        print(f"  member {mid:03d}  success={np.mean(succ):.2f}  dist={np.mean(dists):.3f}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "algo": args.algo,
        "n_test": len(test_ids),
        "episodes_per_member": args.episodes_per_member,
        "test_success_rate": float(np.mean([r["success_rate"] for r in results])),
        "test_mean_final_dist": float(np.mean([r["mean_final_dist"] for r in results])),
        "members": results,
    }
    out = args.out_json or args.checkpoint.parent / f"eval_{args.algo}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n>>> test success rate: {summary['test_success_rate']:.3f}")
    print(f">>> wrote {out}")


if __name__ == "__main__":
    main()
