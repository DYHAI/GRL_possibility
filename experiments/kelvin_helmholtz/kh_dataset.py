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
    """Two-way split: train + test (legacy). Prefer split_members_three_way."""
    train_ids, _, test_ids = split_members_three_way(
        member_ids, n_val=0, n_test=n_test, seed=seed
    )
    return train_ids, test_ids


def split_members_three_way(
    member_ids: list[int],
    n_val: int = 10,
    n_test: int = 20,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Disjoint train / val / test split over member IDs (seeded shuffle)."""
    ids = sorted(member_ids)
    n_hold = n_val + n_test
    if n_hold >= len(ids):
        raise ValueError(
            f"n_val + n_test = {n_hold} must be < n_members = {len(ids)}"
        )
    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    test_ids = sorted(shuffled[:n_test])
    val_ids = sorted(shuffled[n_test : n_test + n_val])
    train_ids = sorted(shuffled[n_test + n_val :])
    return train_ids, val_ids, test_ids


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
    val_ids: list[int],
    test_ids: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
) -> None:
    payload = {
        "data_dir": str(data_dir),
        "seed": seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "var_names": list(VAR_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_split_json(path: Path) -> dict:
    split = json.loads(path.read_text())
    # Legacy splits stored validation members as test_ids only.
    if "val_ids" not in split and "test_ids" in split:
        split["val_ids"] = []
    return split


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


class RandomOneStepDataset(Dataset):
    """Each sample randomly draws one member and one consecutive (t, t+1) pair."""

    def __init__(
        self,
        members: dict[int, MemberSeries],
        mean: np.ndarray,
        std: np.ndarray,
        samples_per_epoch: int,
        seed: int = 0,
    ):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.series_list = list(members.values())
        if not self.series_list:
            raise ValueError("RandomOneStepDataset requires at least one member")
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # idx unused: fresh random pair each call (re-drawn every epoch via DataLoader shuffle)
        series = self.series_list[int(self.rng.integers(0, len(self.series_list)))]
        t = int(self.rng.integers(0, len(series.X) - 1))
        x0, x1 = series.X[t], series.X[t + 1]
        x0n = (x0 - self.mean[:, None, None]) / self.std[:, None, None]
        x1n = (x1 - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(x0n.copy()), torch.from_numpy(x1n.copy())


def count_one_step_pairs(members: dict[int, MemberSeries]) -> int:
    return sum(max(0, len(s.X) - 1) for s in members.values())


def load_pt_dataset(pt_path: Path) -> dict:
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def pt_train_pair_count(data: dict) -> int:
    total = 0
    for mid in data["train_ids"]:
        start, end = data["member_slices"][mid]
        total += max(0, end - start - 1)
    return total


def pt_val_pair_count(data: dict) -> int:
    total = 0
    for mid in data["val_ids"]:
        start, end = data["member_slices"][mid]
        total += max(0, end - start - 1)
    return total


class PtRandomOneStepDataset(Dataset):
    """Random (member, t)->(t+1) from a unified .pt tensor."""

    def __init__(
        self,
        data: dict,
        samples_per_epoch: int,
        seed: int = 0,
        member_ids: list[int] | None = None,
    ):
        self.X = data["X"]
        self.mean = data["mean"].float()
        self.std = data["std"].float()
        self.slices = data["member_slices"]
        ids = member_ids if member_ids is not None else data["train_ids"]
        self.train_ranges = [self.slices[mid] for mid in ids]
        if not self.train_ranges:
            raise ValueError("No train members in .pt dataset")
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.train_ranges[int(self.rng.integers(0, len(self.train_ranges)))]
        t = int(self.rng.integers(start, end - 1))
        x0 = self.X[t].float()
        x1 = self.X[t + 1].float()
        x0n = (x0 - self.mean[:, None, None]) / self.std[:, None, None]
        x1n = (x1 - self.mean[:, None, None]) / self.std[:, None, None]
        return x0n, x1n


class PtOneStepDataset(Dataset):
    """All consecutive pairs from selected members in a unified .pt tensor."""

    def __init__(self, data: dict, member_ids: list[int]):
        self.X = data["X"]
        self.mean = data["mean"].float()
        self.std = data["std"].float()
        self.indices: list[int] = []
        for mid in member_ids:
            start, end = data["member_slices"][mid]
            for t in range(start, end - 1):
                self.indices.append(t)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.indices[idx]
        x0 = self.X[t].float()
        x1 = self.X[t + 1].float()
        x0n = (x0 - self.mean[:, None, None]) / self.std[:, None, None]
        x1n = (x1 - self.mean[:, None, None]) / self.std[:, None, None]
        return x0n, x1n


def members_dict_from_pt(data: dict, member_ids: list[int]) -> dict[int, MemberSeries]:
    """Rebuild MemberSeries dict for eval utilities."""
    out: dict[int, MemberSeries] = {}
    for mid in member_ids:
        start, end = data["member_slices"][mid]
        X = data["X"][start:end].numpy().astype(np.float32)
        times = data["times"][start:end].numpy().astype(np.float32)
        out[mid] = MemberSeries(member_id=mid, X=X, times=times)
    return out


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
