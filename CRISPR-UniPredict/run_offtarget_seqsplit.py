"""
Off-target GUIDE-LEVEL leakage check.

Trains the dedicated off-target model on the guide-level (sequence-level) split
built by scripts/create_offtarget_seqsplit.py, evaluates on its held-out test
guides, and compares to the pair-level headline numbers (AUROC 0.9931 /
AUPRC 0.8315).

A large drop (especially in AUPRC) means the pair-level numbers were inflated by
guide-level leakage (the model memorised guide-specific patterns). A small drop
means the headline result largely holds on unseen guides.

Robust to the PyTorch-on-Windows teardown crash (exit 0xC0000409): success is
detected by a freshly-written checkpoint/JSON, not by the process exit code.

Usage (run from CRISPR-UniPredict/; GPU recommended):
    python scripts/create_offtarget_seqsplit.py     # once, to build the split
    python run_offtarget_seqsplit.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / 'models' / 'checkpoints' / 'off_target_dedicated'

TRAIN = 'data/processed/combined/train_offtarget_seqsplit.csv'
VAL = 'data/processed/combined/val_offtarget_seqsplit.csv'
TEST = 'data/processed/combined/test_offtarget_seqsplit.csv'

# Paper, pair-level split (the number under scrutiny):
PAIR_SPLIT_REF = {'auroc': 0.9931, 'auprc': 0.8315}


def run_expect(cmd, artifact: Path, extra_env=None, label=""):
    """Run subprocess; success = `artifact` (re)written, regardless of exit code
    (PyTorch teardown on Windows can exit 3221226505 after saving)."""
    before = artifact.stat().st_mtime if artifact.exists() else -1.0
    env = os.environ.copy()
    env.setdefault('CRISPR_OFFTARGET_NUM_WORKERS', '0')
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    after = artifact.stat().st_mtime if artifact.exists() else -1.0
    if after > before:
        if r.returncode != 0:
            print(f"  [warn] {label}: exit {r.returncode} but {artifact.name} written "
                  f"-> benign teardown crash, continuing.")
        return
    sys.exit(f"{label} FAILED (exit {r.returncode}, no fresh {artifact.name})")


def main():
    for f in (TRAIN, VAL, TEST):
        if not (ROOT / f).exists():
            sys.exit(f"Missing {f}\n  Run first: python scripts/create_offtarget_seqsplit.py")

    suffix = '_seqsplit_offtarget'
    ckpt = CKPT_DIR / f'best_model{suffix}.pt'
    out = CKPT_DIR / f'test_results{suffix}.json'

    if not ckpt.exists():
        run_expect([sys.executable, 'train_off_target_dedicated.py'], ckpt,
                   {'CRISPR_OFFTARGET_SEED': 42,
                    'CRISPR_OFFTARGET_CKPT_SUFFIX': suffix,
                    'CRISPR_OFFTARGET_DATA_TRAIN': TRAIN,
                    'CRISPR_OFFTARGET_DATA_VAL': VAL},
                   label='train guide-level')
    else:
        print(f"[skip] {ckpt.name} exists -> evaluating existing checkpoint.")

    run_expect([sys.executable, 'evaluate_off_target_dedicated.py',
                '--ckpt', str(ckpt), '--data', TEST, '--out', str(out),
                '--num-workers', '0'], out, label='eval guide-level')

    with open(out) as f:
        res = json.load(f)
    m = res['metrics_threshold_0.5']

    print("\n" + "=" * 64)
    print("OFF-TARGET LEAKAGE CHECK - pair-level vs guide-level split")
    print("=" * 64)
    print(f"{'metric':<8}{'pair-level':>14}{'guide-level':>14}{'delta':>10}")
    for k in ('auroc', 'auprc'):
        ref, got = PAIR_SPLIT_REF[k], m[k]
        print(f"{k:<8}{ref:>14.4f}{got:>14.4f}{got - ref:>+10.4f}")

    drop = PAIR_SPLIT_REF['auprc'] - m['auprc']
    print("\nInterpretation:")
    if drop > 0.15:
        print(f"  AUPRC drops {drop:.3f} on unseen guides -> the pair-level headline was")
        print("  materially inflated by guide-level leakage. Report the guide-level number.")
    elif drop > 0.05:
        print(f"  AUPRC drops {drop:.3f} -> some guide-level optimism; report both numbers.")
    else:
        print(f"  AUPRC drop {drop:.3f} is small -> the headline largely holds out-of-guide.")
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
