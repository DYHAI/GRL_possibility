"""Load KH ensemble 512×512×4 NPZ members and build one-step training pairs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VAR_NAMES = ("rho", "v1", "v2", "p")
N_CHANNELS = 4


@dataclass
class MemberSeries:
    member_id: int
    X: np.ndarray  # (T, 4, H, W) float32
    times: np.ndarray  # (T,)


def discover_members(data_dir: Path) -> list[int]:
    ids: list[int] = []
    for p in sorted(data_dir.glob("member_*")):
        if not p.is_dir():
            continue
        mid = int(p.name.split("_")[1])
        npz = p / f"member_{mid:03d}_512x4.npz"
        if npz.exists():
            ids.append(mid)
    return sorted(ids)


def load_member(data_dir: Path, member_id: int) -> MemberSeries:
    npz_path = data_dir / f"member_{member_id:03d}" / f"member_{member_id:03d}_512x4.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    raw = np.load(npz_path)
    X = np.asarray(raw["X"], dtype=np.float32)
    times = np.asarray(raw["times"], dtype=np.float32)
    if X.ndim != 4 or X.shape[1] != N_CHANNELS:
        raise ValueError(f"Unexpected shape {X.shape} in {npz_path}")
    return MemberSeries(member_id=member_id, X=X, times=times)


def load_members(data_dir: Path, member_ids: list[int] | None = None) -> dict[int, MemberSeries]:
    ids = member_ids if member_ids is not None else discover_members(data_dir)
    return {mid: load_member(data_dir, mid) for mid in ids}


def split_members(
    member_ids: list[int],
    n_test: int = 20,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    ids = sorted(member_ids)
    if n_test >= len(ids):
        raise ValueError(f"n_test={n_test} must be < n_members={len(ids)}")
    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    test_ids = sorted(shuffled[:n_test])
    train_ids = sorted(shuffled[n_test:])
    return train_ids, test_ids


def compute_norm_stats(members: dict[int, MemberSeries]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over all frames in the given members."""
    chunks: list[np.ndarray] = []
    for series in members.values():
        # (T, C, H, W) -> (T*H*W, C)
        x = series.X.transpose(0, 2, 3, 1).reshape(-1, N_CHANNELS)
        chunks.append(x)
    stacked = np.concatenate(chunks, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def save_split_json(
    path: Path,
    data_dir: Path,
    train_ids: list[int],
    test_ids: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
) -> None:
    payload = {
        "data_dir": str(data_dir),
        "seed": seed,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "var_names": list(VAR_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_split_json(path: Path) -> dict:
    return json.loads(path.read_text())


class OneStepDataset(Dataset):
    """All consecutive (t, t+1) pairs from selected members."""

    def __init__(
        self,
        members: dict[int, MemberSeries],
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for series in members.values():
            for t in range(len(series.X) - 1):
                self.pairs.append((series.X[t], series.X[t + 1]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = self.pairs[idx]
        x0n = (x0 - self.mean[:, None, None]) / self.std[:, None, None]
        x1n = (x1 - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(x0n), torch.from_numpy(x1n)


def ensemble_mean_at_frame(
    members: dict[int, MemberSeries],
    frame_idx: int,
) -> np.ndarray | None:
    """Mean field at fixed frame index over members that have enough frames."""
    frames: list[np.ndarray] = []
    for series in members.values():
        if frame_idx < len(series.X):
            frames.append(series.X[frame_idx])
    if not frames:
        return None
    return np.mean(np.stack(frames, axis=0), axis=0).astype(np.float32)
