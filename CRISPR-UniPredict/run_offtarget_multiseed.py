"""
Multi-seed off-target training -> error bars for the dedicated off-target model.

Trains the dedicated off-target model with several seeds and evaluates each on the
pair-level test set, then reports mean +/- std for AUROC / AUPRC / F1 / balanced
accuracy. This replaces the single-seed number in the paper with a proper
mean +/- std (addresses the "single seed, no error bars" concern).

Robust to the PyTorch-on-Windows teardown crash (exit 0xC0000409 /
3221226505): that fault fires *after* the checkpoint/JSON is written, so success
is detected by a freshly-written artifact, not by the process exit code.

Idempotent: seeds whose checkpoint already exists are skipped unless --force.

Usage (run from the CRISPR-UniPredict/ directory; GPU strongly recommended):
    python run_offtarget_multiseed.py                  # train missing seeds, then aggregate
    python run_offtarget_multiseed.py --seeds 42 1 2   # custom seed list
    python run_offtarget_multiseed.py --force          # retrain even if checkpoints exist
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / 'models' / 'checkpoints' / 'off_target_dedicated'
TEST_CSV = 'data/processed/combined/test.csv'
DEFAULT_SEEDS = [42, 1, 2, 3, 4]


MAX_ATTEMPTS = 4  # this is a busy desktop GPU -> retry transient native crashes


def _env(extra=None):
    env = os.environ.copy()
    env.setdefault('CRISPR_OFFTARGET_NUM_WORKERS', '0')  # reliable on Windows native
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _fresh(path: Path, before: float) -> bool:
    return path.exists() and path.stat().st_mtime > before


def train_seed(seed, force):
    """Train one seed, retrying transient mid-training crashes. Completion is
    signalled by training_history{suffix}.json, which the trainer writes only
    after the full loop finishes (a benign teardown crash, exit 0xC0000409,
    happens *after* that write; a mid-training crash leaves no fresh history)."""
    suffix = f'_seed{seed}'
    ckpt = CKPT_DIR / f'best_model{suffix}.pt'
    hist = CKPT_DIR / f'training_history{suffix}.json'
    if hist.exists() and ckpt.exists() and not force:
        print(f"[seed {seed}] already completed -> skip")
        return True
    for attempt in range(1, MAX_ATTEMPTS + 1):
        before = hist.stat().st_mtime if hist.exists() else -1.0
        print(f"\n[seed {seed}] training attempt {attempt}/{MAX_ATTEMPTS}")
        r = subprocess.run([sys.executable, 'train_off_target_dedicated.py'],
                           cwd=str(ROOT),
                           env=_env({'CRISPR_OFFTARGET_SEED': seed,
                                     'CRISPR_OFFTARGET_CKPT_SUFFIX': suffix}))
        if _fresh(hist, before) and ckpt.exists():
            if r.returncode != 0:
                print(f"  [ok] seed {seed} completed (exit {r.returncode} = benign "
                      f"teardown crash after history was written).")
            else:
                print(f"  [ok] seed {seed} completed cleanly.")
            return True
        print(f"  [retry] seed {seed} attempt {attempt} crashed mid-training "
              f"(exit {r.returncode}, no fresh history). Retrying...")
    print(f"  [FAIL] seed {seed} did not complete after {MAX_ATTEMPTS} attempts.")
    return False


def eval_seed(seed):
    suffix = f'_seed{seed}'
    ckpt = CKPT_DIR / f'best_model{suffix}.pt'
    out = CKPT_DIR / f'test_results{suffix}.json'
    for attempt in range(1, MAX_ATTEMPTS + 1):
        before = out.stat().st_mtime if out.exists() else -1.0
        r = subprocess.run([sys.executable, 'evaluate_off_target_dedicated.py',
                            '--ckpt', str(ckpt), '--data', TEST_CSV, '--out', str(out),
                            '--num-workers', '0'], cwd=str(ROOT), env=_env())
        if _fresh(out, before):
            with open(out) as f:
                return json.load(f)
        print(f"  [retry] eval seed {seed} attempt {attempt} produced no result "
              f"(exit {r.returncode}). Retrying...")
    print(f"  [FAIL] eval seed {seed} failed after {MAX_ATTEMPTS} attempts.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    rows = []
    failed = []
    for s in args.seeds:
        if not train_seed(s, args.force):
            failed.append(s)
            continue
        res = eval_seed(s)
        if res is None:
            failed.append(s)
            continue
        m = res['metrics_threshold_0.5']
        rows.append({'seed': s, 'auroc': m['auroc'], 'auprc': m['auprc'],
                     'f1': m['f1'], 'bal_acc': m['bal_acc'],
                     'f1_best': res['metrics_threshold_bestF1']['f1']})
        print(f"  -> seed {s}: AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}")

    if not rows:
        sys.exit("No seeds completed successfully after retries.")
    if failed:
        print(f"\n[warn] seeds that failed after retries and were excluded: {failed}")

    def stat(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    summary = {
        'seeds_requested': args.seeds,
        'seeds_completed': [r['seed'] for r in rows],
        'seeds_failed': failed,
        'per_seed': rows,
        'mean_std': {k: {'mean': stat(k)[0], 'std': stat(k)[1]}
                     for k in ['auroc', 'auprc', 'f1', 'bal_acc', 'f1_best']},
    }
    out = CKPT_DIR / 'offtarget_multiseed_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"OFF-TARGET MULTI-SEED SUMMARY  (n={len(rows)} seeds: {args.seeds})")
    print("=" * 60)
    for k in ['auroc', 'auprc', 'f1', 'bal_acc']:
        mu, sd = stat(k)
        print(f"  {k:<9} {mu:.4f} +/- {sd:.4f}")
    print("\nPer-seed:")
    for r in rows:
        print(f"  seed {r['seed']:<3} AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}  "
              f"F1 {r['f1']:.4f}  BalAcc {r['bal_acc']:.4f}")
    mu, sd = stat('auprc')
    print(f"\nReport as: AUPRC {mu:.4f} +/- {sd:.4f} (mean +/- std over {len(rows)} seeds)")
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
