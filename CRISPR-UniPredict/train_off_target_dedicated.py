"""
Dedicated off-target training.

Beats the joint multitask off-target head by:
  - Using PAIRED (sgRNA, target) input — joint model only saw sgRNA
  - Focal loss with α/γ tuned for the 93:1 imbalance
  - WeightedRandomSampler for balanced batches
  - AdamW + cosine annealing + mixed precision
  - AUPRC-based early stopping (AUROC is too easy on minority class)

Target: AUROC ≥ 0.85, AUPRC ≥ 0.50 (beats CCLMoff 0.82 / 0.48)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from models.off_target_dedicated import (
    OffTargetDedicatedModel, FocalLoss, encode_pair, NUC_TO_IDX,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG = {
    'data_train': 'data/processed/combined/train.csv',
    'data_val':   'data/processed/combined/val.csv',
    'data_test':  'data/processed/combined/test.csv',
    'ckpt_dir':   'models/checkpoints/off_target_dedicated',

    'seq_len':       23,
    'batch_size':    1024,
    'num_workers':   4,
    'epochs':        40,
    'patience':      6,             # early stop on val AUPRC
    'lr':            3e-4,
    'lr_min':        3e-6,
    'weight_decay':  1e-4,

    'focal_alpha':   0.85,          # α: weight on positives (0.5 = balanced, 0.85 = up-weight positives)
    'focal_gamma':   2.0,           # γ: focuses on hard examples
    'pos_oversample_ratio': 0.5,    # WeightedRandomSampler: target ~50% positives per batch

    'seed':          42,
    'use_amp':       True,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('off_target')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in [logging.StreamHandler(), logging.FileHandler(log_path, mode='w', encoding='utf-8')]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------------
# Dataset — encodes on-the-fly per sample
# ---------------------------------------------------------------------------

class OffTargetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 23):
        self.sg = df['sgrna_sequence'].astype(str).str.upper().str.ljust(seq_len, 'N').str[:seq_len].values
        self.tg = df['target_sequence'].astype(str).str.upper().str.ljust(seq_len, 'N').str[:seq_len].values
        self.y  = df['off_target_label'].astype(np.float32).values
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        sg = self.sg[idx]
        tg = self.tg[idx]
        L = self.seq_len

        pair_oh = np.zeros((L, 9), dtype=np.float32)
        pair_lb = np.zeros((L, 2), dtype=np.int64)

        for j in range(L):
            sg_idx = NUC_TO_IDX.get(sg[j], 0)
            tg_idx = NUC_TO_IDX.get(tg[j], 0)
            pair_oh[j, sg_idx] = 1.0
            pair_oh[j, 4 + tg_idx] = 1.0
            if sg[j] != tg[j] and sg[j] != 'N' and tg[j] != 'N':
                pair_oh[j, 8] = 1.0
            pair_lb[j, 0] = sg_idx
            pair_lb[j, 1] = tg_idx

        return (
            torch.from_numpy(pair_oh),
            torch.from_numpy(pair_lb),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def load_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df['task_type'] == 'off_target'].copy()
    df = df.dropna(subset=['sgrna_sequence', 'target_sequence', 'off_target_label'])
    df['off_target_label'] = df['off_target_label'].astype(int)
    return df.reset_index(drop=True)


def make_sampler(labels: np.ndarray, target_pos_ratio: float = 0.5) -> WeightedRandomSampler:
    """
    WeightedRandomSampler to upweight positives.

    target_pos_ratio = 0.5  → equal positives and negatives per epoch
    target_pos_ratio = 0.25 → 25% positives
    """
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0:
        return None
    w_pos = target_pos_ratio / n_pos
    w_neg = (1.0 - target_pos_ratio) / n_neg
    w = np.where(labels == 1, w_pos, w_neg)
    return WeightedRandomSampler(
        weights=torch.from_numpy(w).double(),
        num_samples=n,
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> Dict[str, float]:
    model.eval()
    all_p, all_y = [], []
    for pair_oh, pair_lb, y in loader:
        pair_oh = pair_oh.to(device, non_blocking=True)
        pair_lb = pair_lb.to(device, non_blocking=True)
        if amp:
            with torch.cuda.amp.autocast():
                logits = model(pair_oh, pair_lb).view(-1)
        else:
            logits = model(pair_oh, pair_lb).view(-1)
        all_p.append(torch.sigmoid(logits).float().cpu().numpy())
        all_y.append(y.numpy())
    p = np.concatenate(all_p)
    y = np.concatenate(all_y).astype(int)

    pred_bin = (p >= 0.5).astype(int)
    return {
        'auroc': float(roc_auc_score(y, p)),
        'auprc': float(average_precision_score(y, p)),
        'f1':    float(f1_score(y, pred_bin, zero_division=0)),
        'bal_acc': float(balanced_accuracy_score(y, pred_bin)),
        'n_pos': int(y.sum()),
        'n_neg': int((y == 0).sum()),
        'pred_pos_rate': float(pred_bin.mean()),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    cfg = CFG
    cfg['seed'] = int(os.environ.get('CRISPR_OFFTARGET_SEED', cfg['seed']))
    suffix = os.environ.get('CRISPR_OFFTARGET_CKPT_SUFFIX', '')

    # Optional env overrides (added for multi-seed + guide-level-split runs;
    # defaults unchanged, so existing behaviour is identical when unset).
    cfg['data_train']  = os.environ.get('CRISPR_OFFTARGET_DATA_TRAIN', cfg['data_train'])
    cfg['data_val']    = os.environ.get('CRISPR_OFFTARGET_DATA_VAL',   cfg['data_val'])
    cfg['data_test']   = os.environ.get('CRISPR_OFFTARGET_DATA_TEST',  cfg['data_test'])
    cfg['num_workers'] = int(os.environ.get('CRISPR_OFFTARGET_NUM_WORKERS', cfg['num_workers']))

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    ckpt_dir = Path(cfg['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(ckpt_dir / f'train{suffix}.log')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = cfg['use_amp'] and device == 'cuda'
    log.info(f'Device: {device}  |  AMP: {amp}  |  Seed: {cfg["seed"]}  |  Suffix: "{suffix}"')

    # ── Data ────────────────────────────────────────────────────────────
    log.info('Loading data...')
    df_tr = load_split(cfg['data_train'])
    df_va = load_split(cfg['data_val'])

    log.info(f'Train off-target: {len(df_tr):,}  (pos {int(df_tr["off_target_label"].sum()):,})')
    log.info(f'Val   off-target: {len(df_va):,}  (pos {int(df_va["off_target_label"].sum()):,})')

    ds_tr = OffTargetDataset(df_tr, seq_len=cfg['seq_len'])
    ds_va = OffTargetDataset(df_va, seq_len=cfg['seq_len'])

    sampler = make_sampler(df_tr['off_target_label'].values, cfg['pos_oversample_ratio'])

    dl_tr = DataLoader(ds_tr, batch_size=cfg['batch_size'],
                       sampler=sampler, num_workers=cfg['num_workers'],
                       pin_memory=(device == 'cuda'), drop_last=True,
                       persistent_workers=(cfg['num_workers'] > 0))
    dl_va = DataLoader(ds_va, batch_size=cfg['batch_size'] * 2,
                       shuffle=False, num_workers=cfg['num_workers'],
                       pin_memory=(device == 'cuda'),
                       persistent_workers=(cfg['num_workers'] > 0))

    # ── Model ───────────────────────────────────────────────────────────
    model = OffTargetDedicatedModel(
        seq_len=cfg['seq_len'],
    ).to(device)
    log.info(f'Model params: {model.count_params()["total"]:,}')

    # ── Optim ───────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=cfg['lr_min'])
    loss_fn = FocalLoss(alpha=cfg['focal_alpha'], gamma=cfg['focal_gamma'])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ── Loop ────────────────────────────────────────────────────────────
    best_auprc = -1.0
    best_state = None
    history = []
    patience_left = cfg['patience']

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_batches = 0.0, 0

        for pair_oh, pair_lb, y in dl_tr:
            pair_oh = pair_oh.to(device, non_blocking=True)
            pair_lb = pair_lb.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if amp:
                with torch.cuda.amp.autocast():
                    logits = model(pair_oh, pair_lb).view(-1)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(pair_oh, pair_lb).view(-1)
                loss = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            loss_sum += loss.item()
            n_batches += 1

        sched.step()
        train_loss = loss_sum / max(1, n_batches)
        val_metrics = evaluate(model, dl_va, device, amp)
        dt = time.time() - t0

        history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics,
                        'lr': opt.param_groups[0]['lr']})

        is_best = val_metrics['auprc'] > best_auprc
        marker = ' ← best' if is_best else ''
        log.info(
            f'Ep {epoch:02d}/{cfg["epochs"]}  '
            f'loss={train_loss:.4f}  '
            f'AUROC={val_metrics["auroc"]:.4f}  '
            f'AUPRC={val_metrics["auprc"]:.4f}  '
            f'F1={val_metrics["f1"]:.4f}  '
            f'BalAcc={val_metrics["bal_acc"]:.4f}  '
            f'lr={opt.param_groups[0]["lr"]:.2e}  '
            f'({dt:.1f}s){marker}'
        )

        if is_best:
            best_auprc = val_metrics['auprc']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                'state_dict': best_state,
                'epoch': epoch,
                'val_metrics': val_metrics,
                'cfg': cfg,
            }, ckpt_dir / f'best_model{suffix}.pt')
            patience_left = cfg['patience']
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info(f'Early stopping at epoch {epoch} — best AUPRC = {best_auprc:.4f}')
                break

    # Save history
    with open(ckpt_dir / f'training_history{suffix}.json', 'w') as f:
        json.dump(history, f, indent=2)

    log.info(f'Done. Best val AUPRC = {best_auprc:.4f}')
    return best_auprc


if __name__ == '__main__':
    main()
