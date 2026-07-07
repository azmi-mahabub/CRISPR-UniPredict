"""
5-Seed Ensemble Training for On-Target Model
=============================================
Trains OnTargetDedicatedModel with 5 different random seeds and saves
each best checkpoint as best_model_seed{N}.pt. Use with
evaluate_on_target_tta.py --ensemble for final predictions.

Usage:
    python train_ensemble_on_target.py             # all 5 seeds sequentially
    python train_ensemble_on_target.py 1 3         # only seeds 1 and 3

Each run reuses the same hyperparameters defined in train_on_target_dedicated.CFG.
"""

import os
import sys
import subprocess
from pathlib import Path

SEEDS = [42, 1, 2, 3, 4]
HERE  = Path(__file__).resolve().parent


def main():
    args = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else SEEDS
    print(f"Running ensemble seeds: {args}")

    for i, seed in enumerate(args, 1):
        suffix = '' if seed == 42 else f'_seed{seed}'
        print()
        print("=" * 70)
        print(f"[{i}/{len(args)}]  Training seed={seed}  → best_model{suffix}.pt")
        print("=" * 70)

        env = os.environ.copy()
        env['CRISPR_SEED']        = str(seed)
        env['CRISPR_CKPT_SUFFIX'] = suffix
        env['PYTHONUNBUFFERED']   = '1'

        result = subprocess.run(
            [sys.executable, str(HERE / 'train_on_target_dedicated.py')],
            env=env, cwd=str(HERE),
        )
        if result.returncode != 0:
            print(f"!! Seed {seed} failed with exit code {result.returncode}")
            print(f"   Continuing with remaining seeds...")

    print()
    print("=" * 70)
    print("Ensemble training complete.")
    print(f"Checkpoints in: models/checkpoints/on_target_dedicated/")
    print()
    print("Next step:")
    print("  python evaluate_on_target_tta.py --ensemble")
    print("=" * 70)


if __name__ == '__main__':
    main()
