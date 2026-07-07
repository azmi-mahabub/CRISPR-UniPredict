"""
Generate paper figures from REAL model outputs only.

Produces:
  figures/fig_offtarget_roc.pdf            ROC curve on 311k test pairs
  figures/fig_offtarget_pr.pdf             PR curve on 311k test pairs
  figures/fig_offtarget_proba_hist.pdf     Predicted probability distribution by class
  figures/fig_offtarget_training.pdf       Off-target training curves (loss, AUROC, AUPRC)
  figures/fig_offtarget_compare.pdf        Joint multitask vs dedicated (same split)
  figures/fig_ontarget_perdataset.pdf      Per-dataset Spearman for ensemble+MC8
  figures/fig_ontarget_progression.pdf     Single → MC8 → Ensemble → Ensemble+MC8

All numbers come from running the trained checkpoints on the actual test split,
or from the recorded JSON files. No synthetic data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))

from models.off_target_dedicated import OffTargetDedicatedModel
from train_off_target_dedicated import OffTargetDataset, load_split


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.3,
    'lines.linewidth': 1.4,
})


PROJ = Path(__file__).parent
FIG_DIR = PROJ / 'figures'
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Re-predict off-target test set, save raw predictions, plot
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_off_target() -> tuple:
    print('[off-target] loading checkpoint + test data...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(PROJ / 'models/checkpoints/off_target_dedicated/best_model.pt',
                    map_location=device, weights_only=False)
    model = OffTargetDedicatedModel(seq_len=23).to(device).eval()
    model.load_state_dict(ck['state_dict'])

    df = load_split(str(PROJ / 'data/processed/combined/test.csv'))
    ds = OffTargetDataset(df, seq_len=23)
    dl = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=4,
                    pin_memory=(device == 'cuda'))

    print(f'[off-target] predicting on {len(df):,} pairs...')
    ps, ys = [], []
    for pair_oh, pair_lb, y in dl:
        pair_oh = pair_oh.to(device, non_blocking=True)
        pair_lb = pair_lb.to(device, non_blocking=True)
        if device == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(pair_oh, pair_lb).view(-1)
        else:
            logits = model(pair_oh, pair_lb).view(-1)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
    p = np.concatenate(ps)
    y = np.concatenate(ys).astype(int)

    # cache raw predictions
    np.savez(FIG_DIR / 'offtarget_predictions.npz', p=p, y=y)
    print(f'[off-target] saved predictions: {p.shape[0]} samples, {int(y.sum())} positive')
    return p, y


def plot_offtarget_roc(p, y):
    from sklearn.metrics import roc_auc_score
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(fpr, tpr, color='#1f4e8a', label=f'Dedicated (AUROC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8, label='Random')

    # External baseline reference lines (their reported AUROC, not from our data)
    for name, ref_auc, col in [
        ('CCLMoff (reported)',   0.82, '#c33'),
        ('DeepCRISPR (reported)', 0.79, '#e96'),
        ('CRISPOR (reported)',   0.75, '#aa6'),
    ]:
        ax.scatter([], [], color=col, s=10, label=f'{name}: {ref_auc:.2f}')

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Off-target ROC curve (test split, n=311,044)')
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'fig_offtarget_roc.pdf')
    plt.close(fig)
    print('  -> fig_offtarget_roc.pdf')


def plot_offtarget_pr(p, y):
    from sklearn.metrics import average_precision_score
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    baseline = y.mean()

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(rec, prec, color='#1f4e8a', label=f'Dedicated (AUPRC = {ap:.4f})')
    ax.axhline(baseline, ls='--', color='gray', linewidth=0.8,
               label=f'Random ({baseline:.4f})')

    for name, ref_auprc, col in [
        ('CCLMoff (reported)',   0.48, '#c33'),
        ('DeepCRISPR (reported)', 0.42, '#e96'),
        ('CRISPOR (reported)',   0.35, '#aa6'),
    ]:
        ax.scatter([], [], color=col, s=10, label=f'{name}: {ref_auprc:.2f}')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Off-target precision-recall curve')
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower left', frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'fig_offtarget_pr.pdf')
    plt.close(fig)
    print('  -> fig_offtarget_pr.pdf')


def plot_offtarget_hist(p, y):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bins = np.linspace(0, 1, 51)
    ax.hist(p[y == 0], bins=bins, color='#888', alpha=0.7,
            label=f'Negatives (n={int((y==0).sum()):,})', density=True)
    ax.hist(p[y == 1], bins=bins, color='#1f4e8a', alpha=0.85,
            label=f'Positives (n={int(y.sum()):,})', density=True)
    ax.set_yscale('log')
    ax.set_xlabel('Predicted off-target probability')
    ax.set_ylabel('Density (log scale)')
    ax.set_title('Predicted probability distribution by class')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, which='both')
    fig.savefig(FIG_DIR / 'fig_offtarget_proba_hist.pdf')
    plt.close(fig)
    print('  -> fig_offtarget_proba_hist.pdf')


# ---------------------------------------------------------------------------
# 2. Off-target training curves
# ---------------------------------------------------------------------------

def plot_offtarget_training():
    hist = json.loads((PROJ / 'models/checkpoints/off_target_dedicated/training_history.json').read_text())
    ep = [h['epoch'] for h in hist]
    auroc = [h['auroc'] for h in hist]
    auprc = [h['auprc'] for h in hist]
    loss  = [h['train_loss'] for h in hist]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    ax.plot(ep, loss, color='#1f4e8a', marker='o', markersize=3.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss (focal)')
    ax.set_title('Training loss')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ep, auprc, color='#1f4e8a', marker='o', markersize=3.5,
            label='Val AUPRC')
    ax.plot(ep, auroc, color='#c33', marker='s', markersize=3.5,
            label='Val AUROC')
    best_ep = ep[int(np.argmax(auprc))]
    best_auprc = max(auprc)
    ax.axvline(best_ep, ls=':', color='gray', linewidth=0.8)
    ax.text(best_ep + 0.2, 0.55,
            f'best AUPRC = {best_auprc:.4f}\n(epoch {best_ep})',
            fontsize=7, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation metrics')
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=False, loc='lower right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_offtarget_training.pdf')
    plt.close(fig)
    print('  -> fig_offtarget_training.pdf')


# ---------------------------------------------------------------------------
# 3. Off-target same-split head-to-head comparison
# ---------------------------------------------------------------------------

def plot_offtarget_compare():
    # Joint multitask numbers come from the prior evaluation log (same split)
    joint    = {'AUROC': 0.7288, 'AUPRC': 0.0378, 'F1': 0.1347, 'Bal.Acc': 0.5895}
    dedicated = json.loads((PROJ / 'models/checkpoints/off_target_dedicated/test_results.json').read_text())
    ded = {
        'AUROC':   dedicated['metrics_threshold_0.5']['auroc'],
        'AUPRC':   dedicated['metrics_threshold_0.5']['auprc'],
        'F1':      dedicated['metrics_threshold_bestF1']['f1'],
        'Bal.Acc': dedicated['metrics_threshold_0.5']['bal_acc'],
    }

    metrics = list(joint.keys())
    xs = np.arange(len(metrics))
    w = 0.36

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    b1 = ax.bar(xs - w/2, [joint[m] for m in metrics], width=w,
                color='#aaa', label='Joint multi-task (broken)', edgecolor='black', linewidth=0.4)
    b2 = ax.bar(xs + w/2, [ded[m] for m in metrics], width=w,
                color='#1f4e8a', label='Dedicated paired model', edgecolor='black', linewidth=0.4)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score (test split, n=311,044)')
    ax.set_title('Off-target: joint multi-task vs dedicated (same split)')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(FIG_DIR / 'fig_offtarget_compare.pdf')
    plt.close(fig)
    print('  -> fig_offtarget_compare.pdf')


# ---------------------------------------------------------------------------
# 4. On-target per-dataset Spearman bar chart
# ---------------------------------------------------------------------------

def plot_ontarget_perdataset():
    data = json.loads((PROJ / 'models/checkpoints/on_target_dedicated/test_results_ensemble_mc8.json').read_text())
    per = data['per_dataset']

    # Sort by sample size (descending)
    items = sorted(per.items(), key=lambda kv: -kv[1]['n'])
    names = [k for k, _ in items]
    scores = [v['spearman'] for _, v in items]
    ns = [v['n'] for _, v in items]

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    colors = ['#1f4e8a' if n >= 3000 else ('#7aa6c4' if n >= 1000 else '#ddd') for n in ns]
    bars = ax.bar(range(len(names)), scores, color=colors, edgecolor='black', linewidth=0.4)
    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(i, bar.get_height() + 0.015, f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=7)
        ax.text(i, -0.04, f'n={n:,}', ha='center', va='top', fontsize=6, color='#555')

    ax.axhline(0.72, ls='--', color='#c33', linewidth=0.8,
               label='CRISPR-HNN reported size-wt: 0.72')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Spearman correlation')
    ax.set_title('Per-dataset Spearman (Ensemble + MC8)')
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_ontarget_perdataset.pdf')
    plt.close(fig)
    print('  -> fig_ontarget_perdataset.pdf')


# ---------------------------------------------------------------------------
# 5. On-target progression: single → +MC8 → ensemble → ensemble+MC8
# ---------------------------------------------------------------------------

def plot_ontarget_progression():
    base = PROJ / 'models/checkpoints/on_target_dedicated'
    stages = [
        ('Single',       json.loads((base/'test_results_baseline.json').read_text())),
        ('Single+MC8',   json.loads((base/'test_results_mc8.json').read_text())),
        ('Ensemble[5]',  json.loads((base/'test_results_ensemble.json').read_text())),
        ('Ens+MC8',      json.loads((base/'test_results_ensemble_mc8.json').read_text())),
    ]
    names = [s[0] for s in stages]
    sw = [s[1]['size_weighted_spearman'] for s in stages]
    ov = [s[1]['overall_spearman'] for s in stages]

    xs = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(xs - w/2, sw, width=w, color='#1f4e8a', label='Size-weighted Spearman',
           edgecolor='black', linewidth=0.4)
    ax.bar(xs + w/2, ov, width=w, color='#c33', label='Overall Spearman',
           edgecolor='black', linewidth=0.4)
    for i, v in enumerate(sw):
        ax.text(i - w/2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=6.5)
    for i, v in enumerate(ov):
        ax.text(i + w/2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=6.5)

    ax.axhline(0.72, ls='--', color='#c33', linewidth=0.8, alpha=0.6,
               label='CRISPR-HNN reported size-wt: 0.72')
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel('Spearman correlation')
    ax.set_title('On-target test-set progression')
    ax.set_ylim(0.65, 0.82)
    ax.legend(frameon=False, loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_ontarget_progression.pdf')
    plt.close(fig)
    print('  -> fig_ontarget_progression.pdf')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Output directory: {FIG_DIR}')
    p, y = predict_off_target()
    print('\nPlotting off-target...')
    plot_offtarget_roc(p, y)
    plot_offtarget_pr(p, y)
    plot_offtarget_hist(p, y)
    plot_offtarget_training()
    plot_offtarget_compare()

    print('\nPlotting on-target (from saved JSONs only — no model rerun needed)...')
    plot_ontarget_perdataset()
    plot_ontarget_progression()

    print('\nAll figures written to:', FIG_DIR)


if __name__ == '__main__':
    main()
