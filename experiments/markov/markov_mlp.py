#!/usr/bin/env python3
"""Learn n×n Markov transition matrices with MLP+RMSE vs MLP+cross-entropy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
BATCH_SIZE = 1024


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(n_states: int, lo: float = -1.0, hi: float = 1.0) -> tuple[np.ndarray, int, int, int]:
    states = np.linspace(lo, hi, n_states, dtype=np.float32)
    if n_states <= 10:
        hidden, samples, epochs = 128, 100_000, 120
    elif n_states <= 30:
        hidden, samples, epochs = 256, 200_000, 100
    else:
        hidden, samples, epochs = 512, 500_000, 80
    return states, hidden, samples, epochs


def nearest_state_index(states: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(states - value)))


def build_transition_matrix(n_states: int, *, seed: int, alpha: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.full(n_states, alpha), size=n_states)
    return P


def kl_rows(P_true: np.ndarray, P_pred: np.ndarray, eps: float = 1e-8) -> float:
    p, q = P_true + eps, P_pred + eps
    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


def sample_pairs(P: np.ndarray, n_steps: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_states = P.shape[0]
    rng = np.random.default_rng(seed)
    n_per = max(1, n_steps // n_states)
    cur = np.repeat(np.arange(n_states), n_per)
    nxt = np.array([int(rng.choice(n_states, p=P[s])) for s in cur], dtype=np.int64)
    perm = rng.permutation(len(cur))
    return cur[perm], nxt[perm]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP_RMSE(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = MLP(1, 1, hidden)

    def forward(self, s_cur: torch.Tensor) -> torch.Tensor:
        return self.mlp(s_cur.unsqueeze(-1)).squeeze(-1)


class MLP_Softmax(nn.Module):
    def __init__(self, n_states: int, state_values: np.ndarray, hidden: int):
        super().__init__()
        self.n_states = n_states
        self.mlp = MLP(n_states, n_states, hidden)
        self.register_buffer("state_weights", torch.from_numpy(state_values.copy()).float())

    def forward(self, s_cur_idx: torch.Tensor) -> torch.Tensor:
        oh = F.one_hot(s_cur_idx, self.n_states).float()
        return self.mlp(oh)

    def expected_value(self, s_cur_idx: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.forward(s_cur_idx), dim=-1)
        return (probs * self.state_weights).sum(dim=-1)


def train_rmse(model: MLP_RMSE, loader: DataLoader, epochs: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for s_cur, s_nxt in loader:
            s_cur, s_nxt = s_cur.to(DEVICE), s_nxt.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(model(s_cur), s_nxt).backward()
            opt.step()


def train_softmax(model: MLP_Softmax, loader: DataLoader, epochs: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for s_cur_idx, s_nxt_idx in loader:
            s_cur_idx, s_nxt_idx = s_cur_idx.to(DEVICE), s_nxt_idx.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(s_cur_idx), s_nxt_idx).backward()
            opt.step()


@torch.no_grad()
def scalar_rmse(model: MLP_RMSE, states: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> float:
    pred = model(torch.from_numpy(states[cur]).float().to(DEVICE))
    target = torch.from_numpy(states[nxt]).float().to(DEVICE)
    return torch.sqrt(F.mse_loss(pred, target)).item()


@torch.no_grad()
def scalar_rmse_softmax(model: MLP_Softmax, states: np.ndarray, cur_idx: np.ndarray, nxt: np.ndarray) -> float:
    idx = torch.from_numpy(cur_idx).long().to(DEVICE)
    pred = model.expected_value(idx)
    target = torch.from_numpy(states[nxt]).float().to(DEVICE)
    return torch.sqrt(F.mse_loss(pred, target)).item()


@torch.no_grad()
def transition_matrix_rmse_nearest(model: MLP_RMSE, states: np.ndarray) -> np.ndarray:
    n_states = len(states)
    P = np.zeros((n_states, n_states))
    preds = model(torch.from_numpy(states).float().to(DEVICE)).cpu().numpy()
    for i, mu in enumerate(preds):
        P[i, nearest_state_index(states, mu)] = 1.0
    return P


@torch.no_grad()
def transition_matrix_softmax(model: MLP_Softmax, n_states: int) -> np.ndarray:
    idx = torch.arange(n_states, device=DEVICE)
    return F.softmax(model(idx), dim=-1).cpu().numpy()


def plot_matrices(
    out_dir: Path,
    states: np.ndarray,
    P_true: np.ndarray,
    P_rmse: np.ndarray,
    P_soft: np.ndarray,
    rmse_model: MLP_RMSE,
) -> None:
    n_states = len(states)
    tag = f"{n_states}×{n_states}"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = [f"True P ({tag})", "MLP+RMSE (nearest)", "MLP+softmax"]
    vmax = max(0.08, float(P_true.max()))
    tick_idx = np.linspace(0, n_states - 1, min(6, n_states), dtype=int)

    def tick_labels(indices: np.ndarray) -> list[str]:
        return [f"{states[i]:.2f}" for i in indices]

    for ax, P, title in zip(axes, [P_true, P_rmse, P_soft], titles):
        im = ax.imshow(P, vmin=0, vmax=vmax, cmap="viridis", aspect="equal", origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(tick_idx, tick_labels(tick_idx), fontsize=7, rotation=45)
        ax.set_yticks(tick_idx, tick_labels(tick_idx), fontsize=7)
        ax.set_xlabel("$s_{t+1}$")
        ax.set_ylabel("$s_t$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Markov transition matrices ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "transition_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    with torch.no_grad():
        preds = rmse_model(torch.from_numpy(states).float().to(DEVICE)).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(states, preds, "o-", ms=4, lw=1.2, label="MLP+RMSE $\\mu(s)$")
    ax.plot(states, states, "--", color="gray", label="identity")
    ax.set_xlabel("$s_t$")
    ax.set_ylabel("predicted $\\mu$")
    ax.set_title(f"RMSE scalar predictions ({tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_scalar_predictions.png", dpi=140)
    plt.close(fig)


def run_one(
    n_states: int,
    *,
    seed: int = SEED,
    alpha: float = 1.0,
    lo: float = -1.0,
    hi: float = 1.0,
    epochs: int | None = None,
    samples: int | None = None,
    out_root: Path | None = None,
) -> dict:
    out_dir = (out_root or ROOT / "outputs" / "markov") / f"n{n_states}"
    out_dir.mkdir(parents=True, exist_ok=True)

    states, hidden, default_samples, default_epochs = configure(n_states, lo, hi)
    epochs = epochs or default_epochs
    n_samples = samples or default_samples

    set_seed(seed)
    print(f"\n{'=' * 60}\nMarkov {n_states}×{n_states}  device={DEVICE}\n{'=' * 60}")

    P_true = build_transition_matrix(n_states, seed=seed, alpha=alpha)
    cur_idx, nxt_idx = sample_pairs(P_true, n_samples, seed=seed)
    n_train = int(n_samples * 0.8)
    cur_tr, nxt_tr = cur_idx[:n_train], nxt_idx[:n_train]
    cur_te, nxt_te = cur_idx[n_train:], nxt_idx[n_train:]

    loader_rmse = DataLoader(
        TensorDataset(
            torch.from_numpy(states[cur_tr]).float(),
            torch.from_numpy(states[nxt_tr]).float(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    loader_soft = DataLoader(
        TensorDataset(torch.from_numpy(cur_tr).long(), torch.from_numpy(nxt_tr).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    m_rmse = MLP_RMSE(hidden).to(DEVICE)
    m_soft = MLP_Softmax(n_states, states, hidden).to(DEVICE)

    print(f"Training MLP+RMSE ({epochs} epochs, {n_samples:,} samples)...")
    train_rmse(m_rmse, loader_rmse, epochs=epochs)
    print("Training MLP+softmax...")
    train_softmax(m_soft, loader_soft, epochs=epochs)
    m_rmse.eval()
    m_soft.eval()

    P_rmse = transition_matrix_rmse_nearest(m_rmse, states)
    P_soft = transition_matrix_softmax(m_soft, n_states)

    metrics = {
        "n_states": n_states,
        "state_range": [lo, hi],
        "seed": seed,
        "dirichlet_alpha": alpha,
        "samples": n_samples,
        "epochs": epochs,
        "mlp_rmse": {
            "rmse_test": scalar_rmse(m_rmse, states, cur_te, nxt_te),
            "kl_rows": kl_rows(P_true, P_rmse),
        },
        "mlp_softmax": {
            "rmse_test": scalar_rmse_softmax(m_soft, states, cur_te, nxt_te),
            "kl_rows": kl_rows(P_true, P_soft),
        },
    }

    print(f"{'Model':<16} {'RMSE':>8} {'KL(true||pred)':>14}")
    print("-" * 40)
    print(f"{'MLP+RMSE':<16} {metrics['mlp_rmse']['rmse_test']:8.4f} {metrics['mlp_rmse']['kl_rows']:14.4f}")
    print(f"{'MLP+softmax':<16} {metrics['mlp_softmax']['rmse_test']:8.4f} {metrics['mlp_softmax']['kl_rows']:14.4f}")

    np.savez_compressed(
        out_dir / "transition_matrices.npz",
        P_true=P_true,
        P_mlp_rmse=P_rmse,
        P_mlp_softmax=P_soft,
        states=states,
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_matrices(out_dir, states, P_true, P_rmse, P_soft, m_rmse)
    print(f"Saved to {out_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Markov transition matrix learning experiment")
    p.add_argument("--sizes", type=int, nargs="+", default=[3, 10, 100])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dirichlet-alpha", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "markov")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = {}
    for n in args.sizes:
        summary[str(n)] = run_one(
            n,
            seed=args.seed,
            alpha=args.dirichlet_alpha,
            epochs=args.epochs,
            samples=args.samples,
            out_root=args.out_root,
        )
    summary_path = args.out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll Markov runs complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
