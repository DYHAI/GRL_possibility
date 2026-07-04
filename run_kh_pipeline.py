#!/usr/bin/env python3
"""Run KH U-Net pipeline on Trixi 200-member ensemble data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KH = ROOT / "experiments" / "kelvin_helmholtz"
PY = sys.executable

DEFAULT_DATA = ROOT / "outputs" / "kelvin_helmholtz" / "trixi_ensemble_200"
DEFAULT_OUT = ROOT / "outputs" / "kelvin_helmholtz" / "unet_mse"


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    if not args.skip_train:
        run([
            PY, str(KH / "train_unet.py"),
            "--data-dir", str(args.data_dir),
            "--out-dir", str(args.out_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
        ])

    if not args.skip_eval:
        run([
            PY, str(KH / "eval_unet.py"),
            "--data-dir", str(args.data_dir),
            "--checkpoint", str(args.out_dir / "unet_best.pt"),
            "--split-json", str(args.out_dir / "split.json"),
            "--out-dir", str(args.out_dir / "eval"),
        ])

    if not args.skip_figures:
        run([
            PY, str(KH / "make_paper_figures.py"),
            "--data-dir", str(args.data_dir),
            "--checkpoint", str(args.out_dir / "unet_best.pt"),
            "--split-json", str(args.out_dir / "split.json"),
            "--eval-json", str(args.out_dir / "eval" / "eval_summary.json"),
            "--out-dir", str(ROOT / "outputs" / "kelvin_helmholtz" / "figures"),
        ])

    print("\nDone (Trixi ensemble → U-Net).")
    print(f"  data:     {args.data_dir}")
    print(f"  model:    {args.out_dir / 'unet_best.pt'}")
    print(f"  eval:     {args.out_dir / 'eval'}")
    print(f"  figures:  {ROOT / 'outputs' / 'kelvin_helmholtz' / 'figures'}")


if __name__ == "__main__":
    main()
