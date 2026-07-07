"""
Ensemble Training Script for On-Target Prediction
===================================================
Trains 3 models with different random seeds and saves each.
Ensemble inference averages predictions → typically +0.02-0.03 Spearman.

Usage:
    python train_ensemble.py

Output:
    models/checkpoints/ensemble/seed_42/best_model.pt
    models/checkpoints/ensemble/seed_123/best_model.pt
    models/checkpoints/ensemble/seed_456/best_model.pt
    models/checkpoints/ensemble/ensemble_result.json
"""

import sys, json, logging
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from utils.rna_fm_path import ensure_rna_fm_import_path
ensure_rna_fm_import_path(_ROOT)

from models.on_target_dedicated import OnTargetDedicatedModel
from train_on_target_dedicated import (
    OnTargetDataset,
    compute_norm_stats,
    apply_normalization,
    CombinedRegressionLoss,
    differentiable_spearman_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = dict(
    train_csv      = 'data/processed/combined/train_seqsplit.csv',
    val_csv        = 'data/processed/combined/val_seqsplit.csv',
    rna_fm_path    = 'models/pretrained/rna_fm_t12.pt',
    ensemble_dir   = 'models/checkpoints/ensemble',

    branch_channels        = 64,
    bigru_hidden           = 256,
    embed_dim              = 128,
    fusion_dim             = 256,
    dropout                = 0.15,
    num_transformer_layers = 2,
    num_attention_heads    = 8,

    batch_size   = 512,
    epochs       = 60,
    lr           = 3e-4,
    weight_decay = 1e-4,
    grad_clip    = 1.0,
    patience     = 12,
    warmup_pct   = 0.05,

    mse_w      = 0.4,
    huber_w    = 0.3,
    spearman_w = 0.3,
    huber_delta = 0.3,

    seeds = [42, 123, 456],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0.0
    for oh, lbl, norm_score, raw_score, source in loader:
        oh         = oh.to(DEVICE)
        lbl        = lbl.to(DEVICE)
        norm_score = norm_score.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(oh, lbl)
            loss = criterion(pred, norm_score)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            optimizer.step()
        scheduler.step()
        total_loss += loss.item() * oh.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    """Return (predictions_z, raw_scores, sources) arrays."""
    model.eval()
    preds, raws, srcs = [], [], []
    for oh, lbl, _, raw_score, source in loader:
        p = model(oh.to(DEVICE), lbl.to(DEVICE)).squeeze(-1).cpu().numpy()
        preds.append(p)
        raws.append(raw_score.numpy())
        srcs.extend(source)
    return np.concatenate(preds), np.concatenate(raws), np.array(srcs)


def denormalize(pred_z, sources, norm_stats):
    pred_orig = np.empty_like(pred_z)
    for src, s in norm_stats.items():
        mask = sources == src
        pred_orig[mask] = pred_z[mask] * s['std'] + s['mean']
    unknown = ~np.isin(sources, list(norm_stats.keys()))
    pred_orig[unknown] = pred_z[unknown]
    return pred_orig


def compute_metrics(pred_orig, raw, sources):
    per_ds = {}
    for src in np.unique(sources):
        mask = sources == src
        if mask.sum() >= 4:
            scc, _ = spearmanr(pred_orig[mask], raw[mask])
            per_ds[src] = float(scc)
    src_counts = {s: (sources == s).sum() for s in per_ds}
    total_n = sum(src_counts.values())
    size_wt = sum(per_ds[s] * src_counts[s] for s in per_ds) / total_n
    eq_avg  = np.mean(list(per_ds.values()))
    overall, _ = spearmanr(pred_orig, raw)
    mae  = mean_absolute_error(raw, pred_orig)
    return {
        'spearman_size_wt': float(size_wt),
        'spearman_eq_avg' : float(eq_avg),
        'spearman_overall': float(overall),
        'mae'             : float(mae),
        'per_dataset'     : per_ds,
    }


# ── Train single model ───────────────────────────────────────────────────────

def train_single(seed, train_ds, val_loader, norm_stats, ckpt_path):
    set_seed(seed)
    model = OnTargetDedicatedModel(
        rna_fm_path            = CFG['rna_fm_path'],
        branch_channels        = CFG['branch_channels'],
        bigru_hidden           = CFG['bigru_hidden'],
        embed_dim              = CFG['embed_dim'],
        fusion_dim             = CFG['fusion_dim'],
        dropout                = CFG['dropout'],
        num_transformer_layers = CFG['num_transformer_layers'],
        num_attention_heads    = CFG['num_attention_heads'],
        device                 = str(DEVICE),
    ).to(DEVICE)

    train_loader = DataLoader(
        train_ds, batch_size=CFG['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    criterion = CombinedRegressionLoss(
        CFG['mse_w'], CFG['huber_w'], CFG['spearman_w'], CFG['huber_delta']
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG['lr'],
        weight_decay=CFG['weight_decay'], betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG['lr'],
        total_steps=len(train_loader) * CFG['epochs'],
        pct_start=CFG['warmup_pct'], anneal_strategy='cos',
    )
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    best_scc = -1.0
    patience_counter = 0
    best_metrics = {}

    for epoch in range(1, CFG['epochs'] + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler)
        pred_z, raw, srcs = predict(model, val_loader)
        pred_orig = denormalize(pred_z, srcs, norm_stats)
        m = compute_metrics(pred_orig, raw, srcs)
        scc = m['spearman_size_wt']

        improved = scc > best_scc
        marker   = ' ← best' if improved else ''
        logger.info(
            f"  [seed={seed}] Ep {epoch:02d}/{CFG['epochs']}  "
            f"loss={loss:.4f}  SCC(size-wt)={scc:.4f}  "
            f"SCC(eq-avg)={m['spearman_eq_avg']:.4f}{marker}"
        )

        if improved:
            best_scc     = scc
            best_metrics = m
            patience_counter = 0
            torch.save({'state_dict': model.state_dict(), 'metrics': m,
                        'norm_stats': norm_stats, 'seed': seed}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= CFG['patience']:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    return best_scc, best_metrics


# ── Ensemble inference ───────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_predict(checkpoints, val_loader, norm_stats):
    """Load each checkpoint, predict, average predictions, evaluate."""
    all_pred_z = []
    raw_ref, src_ref = None, None

    for ckpt_path in checkpoints:
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model = OnTargetDedicatedModel(
            rna_fm_path            = CFG['rna_fm_path'],
            branch_channels        = CFG['branch_channels'],
            bigru_hidden           = CFG['bigru_hidden'],
            embed_dim              = CFG['embed_dim'],
            fusion_dim             = CFG['fusion_dim'],
            dropout                = CFG['dropout'],
            num_transformer_layers = CFG['num_transformer_layers'],
            num_attention_heads    = CFG['num_attention_heads'],
            device                 = str(DEVICE),
        ).to(DEVICE)
        model.load_state_dict(ck['state_dict'])
        pred_z, raw, srcs = predict(model, val_loader)
        all_pred_z.append(pred_z)
        if raw_ref is None:
            raw_ref = raw
            src_ref = srcs
        del model

    avg_pred_z   = np.mean(all_pred_z, axis=0)
    avg_pred_orig = denormalize(avg_pred_z, src_ref, norm_stats)
    return compute_metrics(avg_pred_orig, raw_ref, src_ref)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Training ensemble with seeds: {CFG['seeds']}")

    # Load data
    logger.info("Loading data...")
    train_raw = pd.read_csv(CFG['train_csv'], low_memory=False)
    val_raw   = pd.read_csv(CFG['val_csv'],   low_memory=False)
    train_raw.columns = [c.lower() for c in train_raw.columns]
    val_raw.columns   = [c.lower() for c in val_raw.columns]

    train_ot = train_raw[train_raw['task_type'] == 'on_target'].copy()
    val_ot   = val_raw[val_raw['task_type']   == 'on_target'].copy()
    logger.info(f"Train: {len(train_ot):,}  Val: {len(val_ot):,}")

    # Normalization (computed once, shared across all seeds)
    norm_stats = compute_norm_stats(train_ot)
    train_ot_raw = train_ot.copy()
    val_ot_raw   = val_ot.copy()
    train_ot_norm = apply_normalization(train_ot, norm_stats)
    val_ot_norm   = apply_normalization(val_ot,   norm_stats)

    train_ds = OnTargetDataset(train_ot_norm, train_ot_raw)
    val_ds   = OnTargetDataset(val_ot_norm,   val_ot_raw)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'] * 2,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Train each seed
    ensemble_dir = Path(CFG['ensemble_dir'])
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    checkpoints  = []
    single_sccs  = []

    for seed in CFG['seeds']:
        logger.info("=" * 60)
        logger.info(f"Training seed {seed}")
        logger.info("=" * 60)
        ckpt_path = ensemble_dir / f'seed_{seed}' / 'best_model.pt'
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        best_scc, best_m = train_single(seed, train_ds, val_loader, norm_stats, ckpt_path)
        checkpoints.append(ckpt_path)
        single_sccs.append(best_scc)
        logger.info(f"Seed {seed} best SCC(size-wt): {best_scc:.4f}")

    # Ensemble evaluation
    logger.info("=" * 60)
    logger.info("Evaluating ensemble (averaged predictions)...")
    ens_metrics = ensemble_predict(checkpoints, val_loader, norm_stats)

    # Save norm stats for inference
    with open(ensemble_dir / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)

    result = {
        'individual_sccs'  : {f'seed_{s}': float(sc) for s, sc in zip(CFG['seeds'], single_sccs)},
        'ensemble_metrics' : ens_metrics,
    }
    with open(ensemble_dir / 'ensemble_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Final report
    ens_scc = ens_metrics['spearman_size_wt']
    logger.info("=" * 60)
    logger.info("ENSEMBLE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Individual models (size-wt Spearman): "
                + "  ".join(f"seed_{s}={sc:.4f}" for s, sc in zip(CFG['seeds'], single_sccs)))
    logger.info(f"Ensemble  size-weighted Spearman : {ens_scc:.4f}")
    logger.info(f"Ensemble  equal-avg    Spearman  : {ens_metrics['spearman_eq_avg']:.4f}")
    logger.info(f"Ensemble  overall      Spearman  : {ens_metrics['spearman_overall']:.4f}")
    logger.info(f"Ensemble  MAE                    : {ens_metrics['mae']:.4f}")
    logger.info("")
    logger.info("Per-dataset:")
    for src, v in sorted(ens_metrics['per_dataset'].items()):
        logger.info(f"  {src:15s} {v:.4f}")
    logger.info("")
    baselines = {'CRISPR-HNN': 0.72, 'DeepHF': 0.68, 'Seq2Seq': 0.65}
    logger.info("Comparison with baselines:")
    for name, bl in baselines.items():
        diff   = ens_scc - bl
        status = "BEATS" if diff > 0 else "below"
        logger.info(f"  {name:15s} {bl:.2f}  ens={ens_scc:.4f}  ({status} by {abs(diff):.4f})")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {ensemble_dir / 'ensemble_result.json'}")


if __name__ == '__main__':
    main()
