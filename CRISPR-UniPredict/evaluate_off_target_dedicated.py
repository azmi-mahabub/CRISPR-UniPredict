"""
Evaluate the dedicated off-target model on the held-out test set
and compare against the published SOTA baselines.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, precision_recall_curve, roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent))

from models.off_target_dedicated import OffTargetDedicatedModel
from train_off_target_dedicated import OffTargetDataset, load_split


SOTA_BASELINES = {
    'CCLMoff':     {'auroc': 0.82, 'auprc': 0.48},
    'DeepCRISPR':  {'auroc': 0.79, 'auprc': 0.42},
    'CRISPOR':     {'auroc': 0.75, 'auprc': 0.35},
}


@torch.no_grad()
def predict_all(model, loader, device, amp: bool) -> tuple:
    model.eval()
    ps, ys = [], []
    for pair_oh, pair_lb, y in loader:
        pair_oh = pair_oh.to(device, non_blocking=True)
        pair_lb = pair_lb.to(device, non_blocking=True)
        if amp:
            with torch.cuda.amp.autocast():
                logits = model(pair_oh, pair_lb).view(-1)
        else:
            logits = model(pair_oh, pair_lb).view(-1)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys).astype(int)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Pick the threshold that maximizes F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # thr is length n-1
    best = int(np.argmax(f1[:-1])) if len(thr) > 0 else 0
    return float(thr[best] if len(thr) > 0 else 0.5), float(f1[best])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='models/checkpoints/off_target_dedicated/best_model.pt')
    ap.add_argument('--data', default='data/processed/combined/test.csv')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--out', default='models/checkpoints/off_target_dedicated/test_results.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = device == 'cuda'

    print(f'Device: {device}')
    print(f'Loading checkpoint: {args.ckpt}')
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck.get('cfg', {})

    model = OffTargetDedicatedModel(seq_len=cfg.get('seq_len', 23)).to(device)
    model.load_state_dict(ck['state_dict'])

    print(f'Loading test data: {args.data}')
    df = load_split(args.data)
    print(f'Test off-target: {len(df):,}  (pos {int(df["off_target_label"].sum()):,})')

    ds = OffTargetDataset(df, seq_len=cfg.get('seq_len', 23))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device == 'cuda'))

    print('Predicting...')
    p, y = predict_all(model, dl, device, amp)

    # Metrics at default threshold (0.5)
    pred_05 = (p >= 0.5).astype(int)
    auroc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))
    f1_05 = float(f1_score(y, pred_05, zero_division=0))
    bal_05 = float(balanced_accuracy_score(y, pred_05))

    # Metrics at best-F1 threshold
    best_thr, best_f1 = find_best_threshold(y, p)
    pred_best = (p >= best_thr).astype(int)
    bal_best = float(balanced_accuracy_score(y, pred_best))

    # Comparison vs SOTA
    comp = {}
    for name, b in SOTA_BASELINES.items():
        comp[name] = {
            'auroc_diff': auroc - b['auroc'],
            'auprc_diff': auprc - b['auprc'],
            'beats_auroc': auroc > b['auroc'],
            'beats_auprc': auprc > b['auprc'],
            'beats_both':  (auroc > b['auroc']) and (auprc > b['auprc']),
        }

    results = {
        'n_test':    int(len(y)),
        'n_pos':     int(y.sum()),
        'n_neg':     int((y == 0).sum()),
        'metrics_threshold_0.5': {
            'auroc':   auroc,
            'auprc':   auprc,
            'f1':      f1_05,
            'bal_acc': bal_05,
            'pred_pos_rate': float(pred_05.mean()),
        },
        'metrics_threshold_bestF1': {
            'threshold': best_thr,
            'f1':        best_f1,
            'bal_acc':   bal_best,
            'pred_pos_rate': float(pred_best.mean()),
        },
        'vs_sota':           comp,
        'baselines':         SOTA_BASELINES,
        'checkpoint_epoch':  ck.get('epoch'),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Pretty print
    print()
    print('=' * 70)
    print('OFF-TARGET TEST RESULTS')
    print('=' * 70)
    print(f'  n_test = {len(y):,}  (pos {int(y.sum()):,}, neg {int((y==0).sum()):,})')
    print(f'  AUROC    = {auroc:.4f}')
    print(f'  AUPRC    = {auprc:.4f}')
    print(f'  F1@0.5   = {f1_05:.4f}   F1@best({best_thr:.3f}) = {best_f1:.4f}')
    print(f'  BalAcc@0.5 = {bal_05:.4f}   BalAcc@best = {bal_best:.4f}')
    print()
    print('vs SOTA:')
    for name, b in SOTA_BASELINES.items():
        c = comp[name]
        roc_tag = 'BEATS' if c['beats_auroc'] else 'loses'
        prc_tag = 'BEATS' if c['beats_auprc'] else 'loses'
        print(f'  {name:<12}  AUROC {auroc:.4f} vs {b["auroc"]:.2f}  '
              f'({c["auroc_diff"]:+.4f}, {roc_tag})  |  '
              f'AUPRC {auprc:.4f} vs {b["auprc"]:.2f}  '
              f'({c["auprc_diff"]:+.4f}, {prc_tag})')
    print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()
