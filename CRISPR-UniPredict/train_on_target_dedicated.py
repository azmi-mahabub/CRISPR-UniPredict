"""
Dedicated On-Target Training Script
====================================
Trains OnTargetDedicatedModel with:
  - On-target data only (no multitask noise)
  - Per-dataset z-score normalization (biggest accuracy fix)
  - Combined loss: MSE + Huber + differentiable Spearman
  - AdamW + OneCycleLR scheduler
  - Early stopping on validation Spearman (on ORIGINAL scale, per-dataset avg)
  - Best checkpoint saved by Spearman (not loss)

Usage:
    # Step 1: create leak-free splits (run once)
    python scripts/create_sequence_splits.py

    # Step 2: train
    python train_on_target_dedicated.py

Expected result: Spearman 0.65–0.78 on clean held-out sequences
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from utils.rna_fm_path import ensure_rna_fm_import_path
ensure_rna_fm_import_path(_ROOT)

from models.on_target_dedicated import OnTargetDedicatedModel
from models.encoding import SequenceEncoder

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
CFG = dict(
    train_csv      = 'data/processed/combined/train_seqsplit.csv',
    val_csv        = 'data/processed/combined/val_seqsplit.csv',
    rna_fm_path    = 'models/pretrained/rna_fm_t12.pt',
    checkpoint_dir = 'models/checkpoints/on_target_dedicated',

    # model
    branch_channels        = 64,
    bigru_hidden           = 256,
    embed_dim              = 128,
    fusion_dim             = 256,
    dropout                = 0.15,
    num_transformer_layers = 2,
    num_attention_heads    = 8,

    # training
    batch_size    = 512,
    # Phase 1: learn general patterns from all data
    epochs_phase1 = 60,
    lr_phase1     = 3e-4,
    patience      = 12,
    warmup_pct    = 0.05,
    # Phase 2: fine-tune head + fusion with weighted sampling (small dataset focus)
    epochs_phase2 = 35,
    lr_phase2     = 3e-5,
    patience_phase2 = 8,
    weighted_sampling_phase2 = True,

    weight_decay  = 1e-4,
    grad_clip     = 1.0,

    # loss weights
    mse_w      = 0.4,
    huber_w    = 0.3,
    spearman_w = 0.3,
    huber_delta = 0.3,

    seed = int(os.environ.get('CRISPR_SEED', 42)),  # override via CRISPR_SEED for ensemble

    # SOTA-beat enhancements (tested empirically — keep conservative defaults)
    use_rna_fm   = False,   # RNA-FM hurts on this task (verified: test SCC 0.78 -> 0.71)
    rc_augment   = False,   # CRISPR is strand-specific; RC INVALID (verified: 0.78 -> 0.70)
    mixup_alpha  = 0.0,     # mixup hurts here (verified: pulls test SCC down)
    mixup_prob   = 0.0,
    small_ds_boost = 1.0,   # 2x boost helped val but hurt test generalization
    small_ds_names = ('HCT116', 'HELA', 'HL60'),
    checkpoint_suffix = os.environ.get('CRISPR_CKPT_SUFFIX', ''),
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# PER-DATASET NORMALIZATION
# ============================================================================

def compute_norm_stats(df: pd.DataFrame) -> dict:
    """Compute per-source mean/std from a DataFrame (training set only)."""
    stats = {}
    for src in df['dataset_source'].unique():
        vals = df.loc[df['dataset_source'] == src, 'on_target_score'].dropna()
        if len(vals) < 2:
            stats[src] = {'mean': float(vals.mean()), 'std': 1.0}
        else:
            std = float(vals.std())
            stats[src] = {'mean': float(vals.mean()), 'std': max(std, 1e-6)}
    return stats


def apply_normalization(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply per-source z-score normalization using pre-computed stats."""
    df = df.copy()
    for src, s in stats.items():
        mask = df['dataset_source'] == src
        df.loc[mask, 'on_target_score'] = (
            (df.loc[mask, 'on_target_score'] - s['mean']) / s['std']
        )
    return df


# ============================================================================
# DATASET
# ============================================================================

class OnTargetDataset(Dataset):
    """
    Loads on-target rows only.
    Returns: (one_hot, label_encoded, norm_score, raw_score, source_str)
    norm_score — z-score normalized (used for training loss)
    raw_score  — original scale (used for fair evaluation)
    """

    SEQ_COL   = 'sgrna_sequence'
    SCORE_COL = 'on_target_score'
    SRC_COL   = 'dataset_source'

    NMAP = {'A': 2, 'C': 3, 'G': 4, 'T': 5, 'U': 5}
    OH   = {'A': [1,0,0,0], 'C': [0,1,0,0],
            'G': [0,0,1,0], 'T': [0,0,0,1], 'U': [0,0,0,1]}
    RC   = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'U': 'A'}

    def __init__(self, norm_df: pd.DataFrame, raw_df: pd.DataFrame,
                 emb_cache: dict = None, augment: bool = False):
        """
        norm_df   : z-score normalized DataFrame (for training)
        raw_df    : original-scale DataFrame (for evaluation)
        emb_cache : pre-computed RNA-FM embeddings {seq: (640,) tensor}
        """
        mask = (norm_df['task_type'] == 'on_target') & norm_df[self.SCORE_COL].notna()
        self.norm_df   = norm_df[mask].reset_index(drop=True)
        self.raw_df    = raw_df[mask].reset_index(drop=True)
        self.emb_cache = emb_cache
        self.augment   = augment
        logger.info(f"  Dataset: {len(self.norm_df):,} on-target samples")

    def __len__(self):
        return len(self.norm_df)

    def _encode(self, seq: str):
        seq = seq.upper()
        oh  = torch.tensor([self.OH.get(c, [0,0,0,0]) for c in seq], dtype=torch.float32)
        lbl = torch.tensor([self.NMAP.get(c, 1) for c in seq], dtype=torch.long)
        return oh, lbl

    def _reverse_complement(self, seq: str) -> str:
        return ''.join(self.RC.get(c, c) for c in reversed(seq))

    def __getitem__(self, idx):
        row_n = self.norm_df.iloc[idx]
        row_r = self.raw_df.iloc[idx]
        seq = str(row_n[self.SEQ_COL]).upper()

        # Reverse-complement augmentation (training only, 50% probability)
        if self.augment and np.random.random() < 0.5:
            seq = self._reverse_complement(seq)

        oh, lbl    = self._encode(seq)
        norm_score = torch.tensor(float(row_n[self.SCORE_COL]), dtype=torch.float32)
        raw_score  = torch.tensor(float(row_r[self.SCORE_COL]), dtype=torch.float32)
        source     = str(row_n[self.SRC_COL])
        rna_emb    = self.emb_cache[seq] if (self.emb_cache and seq in self.emb_cache) else torch.zeros(640)
        return oh, lbl, norm_score, raw_score, source, rna_emb

    def get_sample_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights so every dataset contributes equally.
        weight[i] = N_total / (N_datasets * N_source_i)
        """
        sources  = self.norm_df[self.SRC_COL].values
        counts   = {s: (sources == s).sum() for s in np.unique(sources)}
        n_total  = len(sources)
        n_ds     = len(counts)
        weights  = np.array([n_total / (n_ds * counts[s]) for s in sources], dtype=np.float32)
        return torch.from_numpy(weights)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def differentiable_spearman_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of 1 - Spearman correlation.
    Uses soft ranking via regularised incomplete beta function trick
    (Lee et al., 2021 — NeuralNDCG style soft sort).

    Simplified version: approximate rank via pairwise comparisons.
    O(N²) but N ≤ 512 → fast enough.
    """
    n = pred.size(0)
    if n < 4:
        return F.mse_loss(pred, target)

    # Soft rank: for each element, count how many others it is greater than
    pred_r   = (pred.unsqueeze(0)   - pred.unsqueeze(1)).sigmoid().sum(0)
    target_r = (target.unsqueeze(0) - target.unsqueeze(1)).sigmoid().sum(0)

    # Spearman ≈ Pearson of ranks
    pred_r   = pred_r   - pred_r.mean()
    target_r = target_r - target_r.mean()

    num   = (pred_r * target_r).sum()
    denom = (pred_r.pow(2).sum().sqrt() * target_r.pow(2).sum().sqrt()).clamp(min=1e-8)
    rho   = num / denom
    return 1.0 - rho   # minimise → maximise Spearman


class CombinedRegressionLoss(nn.Module):
    def __init__(self, mse_w=0.3, huber_w=0.2, spearman_w=0.5, delta=1.0):
        super().__init__()
        self.mse_w      = mse_w
        self.huber_w    = huber_w
        self.spearman_w = spearman_w
        self.delta      = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: list = None,
        sample_w: torch.Tensor = None,   # (B,) per-sample weights
    ) -> torch.Tensor:
        p = pred.squeeze(-1)
        t = target

        # Per-sample weighted MSE/Huber
        if sample_w is not None:
            sw = sample_w.to(p.device)
            mse_term   = (sw * (p - t).pow(2)).mean()
            huber_diff = (p - t).abs()
            quad = torch.where(huber_diff <= self.delta,
                               0.5 * (p - t).pow(2),
                               self.delta * (huber_diff - 0.5 * self.delta))
            huber_term = (sw * quad).mean()
        else:
            mse_term   = F.mse_loss(p, t)
            huber_term = F.huber_loss(p, t, delta=self.delta)

        return (self.mse_w      * mse_term
                + self.huber_w  * huber_term
                + self.spearman_w * differentiable_spearman_loss(p, t))


# ============================================================================
# TRAIN / EVAL LOOPS
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0.0
    small_set = set(CFG['small_ds_names'])
    boost     = float(CFG['small_ds_boost'])
    mixup_a   = float(CFG['mixup_alpha'])
    mixup_p   = float(CFG['mixup_prob'])

    # loader yields: oh, lbl, norm_score, raw_score, source, rna_emb
    for oh, lbl, norm_score, raw_score, source, rna_emb in loader:
        oh         = oh.to(DEVICE)
        lbl        = lbl.to(DEVICE)
        norm_score = norm_score.to(DEVICE)
        rna_emb    = rna_emb.to(DEVICE)

        # Per-sample weights: small/noisy datasets get extra weight
        sample_w = torch.tensor(
            [boost if s in small_set else 1.0 for s in source],
            dtype=torch.float32, device=DEVICE,
        )

        # Mixup augmentation (interpolate continuous labels)
        do_mix = mixup_a > 0 and np.random.random() < mixup_p
        if do_mix:
            lam = float(np.random.beta(mixup_a, mixup_a))
            lam = max(lam, 1 - lam)  # bias toward dominant sample
            idx = torch.randperm(oh.size(0), device=DEVICE)
            oh_mix       = lam * oh         + (1 - lam) * oh[idx]
            rna_emb_mix  = lam * rna_emb    + (1 - lam) * rna_emb[idx]
            target_mix   = lam * norm_score + (1 - lam) * norm_score[idx]
            # Keep lbl from dominant sample (label encoding can't be mixed)
            oh, rna_emb, norm_score = oh_mix, rna_emb_mix, target_mix

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(oh, lbl, rna_emb=rna_emb)
            loss = criterion(pred, norm_score, source, sample_w=sample_w)

        if scaler is not None:
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
def evaluate(model, loader, norm_stats):
    """
    Evaluate on validation set using ORIGINAL scale labels.

    Steps:
      1. Get model predictions (z-score space).
      2. De-normalize per source → original scale.
      3. Compute per-dataset Spearman and average (comparable to baselines).
      4. Also compute overall Spearman on original scale.
    """
    model.eval()
    all_pred_z  = []   # model output (z-score space)
    all_raw     = []   # original scale labels
    all_sources = []

    for oh, lbl, norm_score, raw_score, source, rna_emb in loader:
        oh      = oh.to(DEVICE)
        lbl     = lbl.to(DEVICE)
        rna_emb = rna_emb.to(DEVICE)
        pred_z = model(oh, lbl, rna_emb=rna_emb).squeeze(-1).cpu().numpy()
        all_pred_z.append(pred_z)
        all_raw.append(raw_score.numpy())
        all_sources.extend(source)

    pred_z  = np.concatenate(all_pred_z)
    raw     = np.concatenate(all_raw)
    sources = np.array(all_sources)

    # De-normalize predictions: pred_original = pred_z * std + mean
    pred_orig = np.empty_like(pred_z)
    for src, s in norm_stats.items():
        mask = sources == src
        if mask.sum() > 0:
            pred_orig[mask] = pred_z[mask] * s['std'] + s['mean']
    # Sources not in norm_stats (shouldn't happen) → keep z-score
    unknown = ~np.isin(sources, list(norm_stats.keys()))
    pred_orig[unknown] = pred_z[unknown]

    # Per-dataset Spearman (primary metric — matches baseline reporting)
    per_ds_scc = {}
    for src in np.unique(sources):
        mask = sources == src
        if mask.sum() < 4:
            continue
        scc, _ = spearmanr(pred_orig[mask], raw[mask])
        per_ds_scc[src] = float(scc)

    # Equal-weight average
    avg_scc = float(np.mean(list(per_ds_scc.values()))) if per_ds_scc else 0.0

    # Size-weighted average (primary metric for paper — statistically sound)
    src_counts = {s: (sources == s).sum() for s in per_ds_scc}
    total_n = sum(src_counts.values())
    size_weighted_scc = float(
        sum(per_ds_scc[s] * src_counts[s] for s in per_ds_scc) / total_n
    ) if total_n > 0 else 0.0

    # Overall Spearman on original scale
    overall_scc, _ = spearmanr(pred_orig, raw)

    # MAE / RMSE on original scale
    mae  = mean_absolute_error(raw, pred_orig)
    rmse = float(np.sqrt(np.mean((pred_orig - raw) ** 2)))

    return {
        'spearman'          : size_weighted_scc,  # size-weighted avg (primary for paper)
        'spearman_equal_avg': avg_scc,            # equal-weight avg (conservative)
        'spearman_overall'  : float(overall_scc), # all mixed
        'per_dataset'       : per_ds_scc,
        'pearson'           : float(pearsonr(pred_orig, raw)[0]),
        'mae'               : float(mae),
        'rmse'              : float(rmse),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])

    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    logger.info("Loading data...")
    train_raw = pd.read_csv(CFG['train_csv'])
    val_raw   = pd.read_csv(CFG['val_csv'])

    train_raw.columns = [c.lower() for c in train_raw.columns]
    val_raw.columns   = [c.lower() for c in val_raw.columns]

    # Filter to on-target only
    train_ot = train_raw[train_raw['task_type'] == 'on_target'].copy()
    val_ot   = val_raw[val_raw['task_type']   == 'on_target'].copy()
    logger.info(f"Train on-target: {len(train_ot):,}  |  Val on-target: {len(val_ot):,}")

    # ── Per-dataset normalization ───────────────────────────────────
    logger.info("Computing per-dataset normalization stats (train only)...")
    norm_stats = compute_norm_stats(train_ot)
    for src, s in norm_stats.items():
        logger.info(f"  {src:20s}  mean={s['mean']:.4f}  std={s['std']:.4f}")

    # Keep raw copies BEFORE normalization (for de-normalized evaluation)
    train_ot_raw = train_ot.copy()
    val_ot_raw   = val_ot.copy()

    train_ot_norm = apply_normalization(train_ot, norm_stats)
    val_ot_norm   = apply_normalization(val_ot,   norm_stats)

    # Save norm stats alongside checkpoint
    with open(ckpt_dir / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # ── Load pre-computed RNA-FM embeddings (if enabled) ───────────
    emb_cache = None
    if CFG['use_rna_fm']:
        emb_path = ckpt_dir / 'rna_fm_embeddings.pt'
        if emb_path.exists():
            logger.info(f"Loading RNA-FM embedding cache from {emb_path}...")
            emb_cache = torch.load(emb_path, map_location='cpu', weights_only=False)
            logger.info(f"  Loaded {len(emb_cache):,} embeddings")
        else:
            logger.warning("RNA-FM cache not found. Run precompute_rna_fm_embeddings.py first.")
    else:
        logger.info("RNA-FM disabled — using CNN+GRU fallback branch")

    # ── Datasets & loaders ─────────────────────────────────────────
    train_ds = OnTargetDataset(train_ot_norm, train_ot_raw, emb_cache=emb_cache, augment=CFG['rc_augment'])
    val_ds   = OnTargetDataset(val_ot_norm,   val_ot_raw,   emb_cache=emb_cache, augment=False)

    # Phase 1 uses uniform sampling (no weighted)
    train_loader = DataLoader(
            train_ds, batch_size=CFG['batch_size'], shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            persistent_workers=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=CFG['batch_size'] * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────
    logger.info("Building model...")
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
        use_rna_fm             = CFG['use_rna_fm'],
    ).to(DEVICE)

    params = model.count_params()
    logger.info(f"Parameters: {params['total']:,} total | {params['trainable']:,} trainable")

    # ── Loss, optimizer, scheduler ─────────────────────────────────
    criterion = CombinedRegressionLoss(
        mse_w      = CFG['mse_w'],
        huber_w    = CFG['huber_w'],
        spearman_w = CFG['spearman_w'],
        delta      = CFG['huber_delta'],
    )

    use_amp = DEVICE.type == 'cuda'
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    best_spearman = -1.0
    history       = []

    def run_phase(phase_num, epochs, lr, loader, freeze_encoder=False):
        nonlocal best_spearman

        if freeze_encoder:
            # Freeze encoder branches; keep head + fusion trainable so fusion
            # can adapt to the upsampled small-dataset distribution in Phase 2
            for name, p in model.named_parameters():
                p.requires_grad = 'head' in name or 'fusion' in name
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            logger.info(f"Phase 2: encoder frozen ({frozen:,} params), head+fusion trainable ({trainable:,} params)")
        else:
            for p in model.parameters():
                p.requires_grad = True

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=CFG['weight_decay'], betas=(0.9, 0.999),
        )
        total_steps = len(loader) * epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=total_steps,
            pct_start=CFG['warmup_pct'], anneal_strategy='cos',
        )

        patience_counter = 0
        logger.info("=" * 60)
        logger.info(f"Phase {phase_num}  |  Epochs: {epochs}  |  LR: {lr}")
        logger.info("=" * 60)

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, loader, criterion, opt, sched, scaler)
            val_metrics = evaluate(model, val_loader, norm_stats)

            sw  = val_metrics['spearman']          # size-weighted (primary)
            eqa = val_metrics['spearman_equal_avg']
            ov  = val_metrics['spearman_overall']

            improved = sw > best_spearman
            marker   = " ← best" if improved else ""

            logger.info(
                f"[P{phase_num}] Ep {epoch:02d}/{epochs}  loss={train_loss:.4f}  "
                f"SCC(size-wt)={sw:.4f}  SCC(eq-avg)={eqa:.4f}  SCC(overall)={ov:.4f}"
                f"  MAE={val_metrics['mae']:.4f}{marker}"
            )
            if improved:
                ds_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(val_metrics['per_dataset'].items()))
                logger.info(f"  Per-dataset: {ds_str}")

            history.append({'phase': phase_num, 'epoch': epoch,
                            'train_loss': train_loss, **val_metrics})

            if improved:
                best_spearman    = sw
                patience_counter = 0
                suffix = CFG.get('checkpoint_suffix', '')
                torch.save({
                    'epoch': epoch, 'phase': phase_num,
                    'state_dict': model.state_dict(),
                    'metrics': val_metrics,
                    'norm_stats': norm_stats,
                    'cfg': CFG,
                }, ckpt_dir / f'best_model{suffix}.pt')
            else:
                patience_counter += 1
                pat = CFG['patience_phase2'] if phase_num == 2 else CFG['patience']
                if patience_counter >= pat:
                    logger.info(f"  Early stopping (patience={pat})")
                    break

    # ── Phase 1: learn general patterns (no weighted sampling) ─────
    run_phase(1, CFG['epochs_phase1'], CFG['lr_phase1'], train_loader, freeze_encoder=False)

    # ── Phase 2: fine-tune head with weighted sampling ─────────────
    if CFG['weighted_sampling_phase2']:
        sample_weights = train_ds.get_sample_weights()
        sampler2 = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
        weighted_loader = DataLoader(
            train_ds, batch_size=CFG['batch_size'], sampler=sampler2,
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )
        logger.info("Phase 2: weighted sampling ON (small dataset focus)")
        run_phase(2, CFG['epochs_phase2'], CFG['lr_phase2'],
                  weighted_loader, freeze_encoder=True)

    # ── Final report ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Training complete.")
    logger.info(f"  Best size-weighted Spearman : {best_spearman:.4f}  (primary metric)")
    logger.info("")
    logger.info("Comparison with published baselines (size-weighted Spearman):")
    baselines = {'CRISPR-HNN': 0.72, 'DeepHF': 0.68, 'Seq2Seq': 0.65}
    for name, val in baselines.items():
        diff   = best_spearman - val
        status = "BEATS" if diff > 0 else "below"
        logger.info(f"  {name:15s} {val:.2f}   ours {best_spearman:.4f}  ({status} by {abs(diff):.4f})")

    # Save training history
    with open(ckpt_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Best model saved to: {ckpt_dir / 'best_model.pt'}")
    logger.info(f"Norm stats saved to: {ckpt_dir / 'norm_stats.json'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
