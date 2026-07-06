"""Lazy loader for KH ensemble velocity fields (v1, v2) with optional downsampling."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.kelvin_helmholtz.kh_dataset import discover_members, load_member  # noqa: E402

V1, V2 = 1, 2


@dataclass
class FlowSnapshot:
    member_id: int
    frame: int
    v1: np.ndarray  # (H, W)
    v2: np.ndarray  # (H, W)
    grid_n: int


def _downsample_field(field: np.ndarray, grid_n: int) -> np.ndarray:
    if field.shape[-1] == grid_n:
        return field.astype(np.float32)
    t, h, w = field.shape[0], grid_n, grid_n
    src_h, src_w = field.shape[1], field.shape[2]
    yi = np.linspace(0, src_h - 1, h).astype(np.int32)
    xi = np.linspace(0, src_w - 1, w).astype(np.int32)
    out = np.empty((t, h, w), dtype=np.float32)
    for i in range(t):
        out[i] = field[i][np.ix_(yi, xi)]
    return out


class KHFlowCache:
    """Cache velocity fields per member; load NPZ on first access only."""

    def __init__(self, data_dir: Path, member_ids: list[int], grid_n: int = 64):
        self.data_dir = Path(data_dir)
        self.member_ids = sorted(member_ids)
        self.grid_n = grid_n
        self._v1: dict[int, np.ndarray] = {}
        self._v2: dict[int, np.ndarray] = {}
        self._n_frames: dict[int, int] = {}

    def n_frames(self, member_id: int) -> int:
        if member_id not in self._n_frames:
            self._load_member(member_id)
        return self._n_frames[member_id]

    def _load_member(self, member_id: int) -> None:
        series = load_member(self.data_dir, member_id)
        self._v1[member_id] = _downsample_field(series.X[:, V1], self.grid_n)
        self._v2[member_id] = _downsample_field(series.X[:, V2], self.grid_n)
        self._n_frames[member_id] = len(series.X)

    def sample_velocity(self, member_id: int, frame: int, x: float, y: float) -> tuple[float, float]:
        """Bilinear sample flow at physical coords (x,y) in [-1,1]^2."""
        if member_id not in self._v1:
            self._load_member(member_id)
        frame = int(np.clip(frame, 0, self._n_frames[member_id] - 1))
        g = self.grid_n
        # physical [-1,1] -> index [0, g-1]
        fx = (x + 1.0) * 0.5 * (g - 1)
        fy = (y + 1.0) * 0.5 * (g - 1)
        ix = int(np.clip(np.floor(fx), 0, g - 2))
        iy = int(np.clip(np.floor(fy), 0, g - 2))
        tx = fx - ix
        ty = fy - iy
        v1 = self._v1[member_id][frame]
        v2 = self._v2[member_id][frame]
        u = (
            (1 - tx) * (1 - ty) * v1[iy, ix]
            + tx * (1 - ty) * v1[iy, ix + 1]
            + (1 - tx) * ty * v1[iy + 1, ix]
            + tx * ty * v1[iy + 1, ix + 1]
        )
        v = (
            (1 - tx) * (1 - ty) * v2[iy, ix]
            + tx * (1 - ty) * v2[iy, ix + 1]
            + (1 - tx) * ty * v2[iy + 1, ix]
            + tx * ty * v2[iy + 1, ix + 1]
        )
        return float(u), float(v)

    def evict(self, member_id: int) -> None:
        self._v1.pop(member_id, None)
        self._v2.pop(member_id, None)
        self._n_frames.pop(member_id, None)


def default_data_dir() -> Path:
    candidates = [
        ROOT / "outputs/kelvin_helmholtz/trixi_ensemble_200",
        Path("/home/ding/桌面/Interests/200_ensemble"),
    ]
    for p in candidates:
        if discover_members(p):
            return p
    return candidates[0]
