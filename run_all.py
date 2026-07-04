#!/usr/bin/env python3
"""Run Markov experiments (KH uses separate Trixi pipeline — see README)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    py = sys.executable
    markov = ROOT / "experiments" / "markov" / "markov_mlp.py"

    run([py, str(markov), "--sizes", "3", "10", "100"], ROOT)

    print("\nMarkov experiments finished.")
    print(f"  outputs: {ROOT / 'outputs' / 'markov'}")
    print("\nFor KH (Trixi): bash experiments/kelvin_helmholtz/trixi/run_gif_3s.sh")


if __name__ == "__main__":
    main()
