#!/usr/bin/env python3
"""Train PPO / TD3 on AUV-KH navigation (180 train members)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.AUV_KH.auv_kh_env import AUVConfig, AUVKHNavEnv
from experiments.AUV_KH.kh_flow_cache import KHFlowCache, default_data_dir
from experiments.kelvin_helmholtz.kh_dataset import discover_members, split_members_three_way


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO or TD3 on KH flow navigation")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/auv_kh/rl",
    )
    parser.add_argument("--algo", choices=("ppo", "td3"), default="ppo")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-n", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=20)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO, TD3
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        raise SystemExit(
            "Install RL deps: pip install gymnasium stable-baselines3\n" + str(e)
        ) from e

    data_dir = args.data_dir or default_data_dir()
    member_ids = discover_members(data_dir)
    if len(member_ids) < args.n_test + 1:
        raise SystemExit(f"Need >= {args.n_test + 1} members, found {len(member_ids)}")

    train_ids, _, test_ids = split_members_three_way(
        member_ids, n_val=0, n_test=args.n_test, seed=args.seed
    )
    print(f">>> data={data_dir}")
    print(f">>> train={len(train_ids)}  test={len(test_ids)}  algo={args.algo}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "seed": args.seed,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "train_ids": train_ids,
                "test_ids": test_ids,
            },
            indent=2,
        )
    )

    cache = KHFlowCache(data_dir, train_ids, grid_n=args.grid_n)

    def _make():
        return Monitor(AUVKHNavEnv(cache, train_ids, config=AUVConfig(grid_n=args.grid_n), seed=args.seed))

    env = make_vec_env(_make, n_envs=1, seed=args.seed)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
        )
    else:
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
        )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    ckpt = args.out_dir / f"{args.algo}_auv_kh"
    model.save(str(ckpt))
    print(f">>> saved {ckpt}.zip")


if __name__ == "__main__":
    main()
