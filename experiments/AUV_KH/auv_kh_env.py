"""
Gymnasium env: AUV navigates top-left -> bottom-right in a KH flow field.

Physics (toy model):
  - control thrust + local flow advection - linear drag
  - flow (v1, v2) from ensemble member NPZ, time-varying frames
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from experiments.AUV_KH.kh_flow_cache import KHFlowCache


@dataclass
class AUVConfig:
    start: tuple[float, float] = (-0.85, 0.85)   # top-left in (x,y), y up
    goal: tuple[float, float] = (0.85, -0.85)    # bottom-right
    goal_radius: float = 0.08
    dt: float = 0.2
    max_steps: int = 80
    thrust_scale: float = 0.35
    flow_scale: float = 1.0
    drag: float = 0.15
    max_speed: float = 1.2
    step_penalty: float = 0.01
    goal_reward: float = 10.0
    progress_scale: float = 2.0
    grid_n: int = 64


class AUVKHNavEnv(gym.Env):
    """Navigate an AUV through stochastic KH velocity fields."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        flow_cache: KHFlowCache,
        member_ids: list[int],
        *,
        config: AUVConfig | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.flow = flow_cache
        self.member_ids = list(member_ids)
        if not self.member_ids:
            raise ValueError("member_ids must be non-empty")
        self.cfg = config or AUVConfig()
        self.rng = np.random.default_rng(seed)

        # [rel_gx, rel_gy, vx, vy, u_flow, v_flow, x, y]
        high = np.array([2, 2, 2, 2, 3, 3, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.member_id = 0
        self.frame = 0
        self.step_i = 0
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = np.array(self.cfg.goal, dtype=np.float32)

    def _sample_episode(self) -> None:
        self.member_id = int(self.rng.choice(self.member_ids))
        nf = self.flow.n_frames(self.member_id)
        # leave room for max_steps frame advances
        t0_max = max(0, nf - self.cfg.max_steps - 1)
        self.frame = int(self.rng.integers(0, t0_max + 1)) if t0_max > 0 else 0
        self.pos = np.array(self.cfg.start, dtype=np.float32)
        self.pos += self.rng.uniform(-0.03, 0.03, size=2).astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.step_i = 0

    def _obs(self) -> np.ndarray:
        u, v = self.flow.sample_velocity(self.member_id, self.frame, float(self.pos[0]), float(self.pos[1]))
        rel = self.goal - self.pos
        return np.array(
            [rel[0], rel[1], self.vel[0], self.vel[1], u, v, self.pos[0], self.pos[1]],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._sample_episode()
        return self._obs(), {"member_id": self.member_id, "frame": self.frame}

    def step(self, action: np.ndarray):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        u_flow, v_flow = self.flow.sample_velocity(
            self.member_id, self.frame, float(self.pos[0]), float(self.pos[1])
        )

        dist_before = float(np.linalg.norm(self.goal - self.pos))

        # thrust + flow - drag
        self.vel = (1.0 - cfg.drag) * self.vel + cfg.thrust_scale * action * cfg.dt
        self.vel[0] += cfg.flow_scale * u_flow * cfg.dt
        self.vel[1] += cfg.flow_scale * v_flow * cfg.dt
        speed = float(np.linalg.norm(self.vel))
        if speed > cfg.max_speed:
            self.vel *= cfg.max_speed / speed

        self.pos = self.pos + self.vel * cfg.dt
        self.pos = np.clip(self.pos, -1.0, 1.0)

        self.frame += 1
        self.step_i += 1

        dist_after = float(np.linalg.norm(self.goal - self.pos))
        reward = cfg.progress_scale * (dist_before - dist_after) - cfg.step_penalty

        reached = dist_after <= cfg.goal_radius
        terminated = reached
        truncated = self.step_i >= cfg.max_steps
        if reached:
            reward += cfg.goal_reward

        return self._obs(), float(reward), terminated, truncated, {
            "member_id": self.member_id,
            "dist": dist_after,
            "success": reached,
        }


def make_env(
    data_dir: Path,
    member_ids: list[int],
    grid_n: int = 64,
    seed: int = 0,
) -> AUVKHNavEnv:
    cache = KHFlowCache(data_dir, member_ids, grid_n=grid_n)
    return AUVKHNavEnv(cache, member_ids, seed=seed)
