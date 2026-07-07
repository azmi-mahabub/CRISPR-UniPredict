"""
Two-Phase Training Script for On-Target Improvement
Phase 1: Joint training (epochs 1-30)
Phase 2: On-target focused training (epochs 31-60)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from utils.rna_fm_path import ensure_rna_fm_import_path

ensure_rna_fm_import_path(_ROOT)

from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder
from utils.preprocessing.dataset import CRISPRDataset
from utils.losses import MultiTaskLoss
from utils.evaluation.metrics import MetricsCalculator
from configs.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_PATH = 'configs/model_config.yaml'
EXPERIMENT_NAME = 'exp_003_on_target_focused'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATASET LOADING
# ============================================================================
def load_data():
    """Load train and validation datasets"""
    logger.info("Loading datasets...")
    
    train_df = pd.read_csv('data/processed/combined/train.csv')
    val_df = pd.read_csv('data/processed/combined/val.csv')
    train_df.columns = [str(c).lower() for c in train_df.columns]
    val_df.columns = [str(c).lower() for c in val_df.columns]

    logger.info(f"Train: {len(train_df):,} samples")
    logger.info(f"Val: {len(val_df):,} samples")

    seq_cols = ['sgrna_sequence', 'guide_seq', 'sgrna', 'sequence', 'grna']
    seq_col = next((c for c in seq_cols if c in train_df.columns), train_df.columns[0])
    target_cols = ['target_sequence', 'target', 'target_seq']
    target_col = next((c for c in target_cols if c in train_df.columns), None)
    on_cols = ['on_target_score', 'on_target', 'efficiency']
    on_col = next((c for c in on_cols if c in train_df.columns), None)
    off_cols = ['off_target_label', 'off_target', 'label']
    off_col = next((c for c in off_cols if c in train_df.columns), None)

    if target_col is None:
        logger.warning("No target column found; RNA-FM branch C will use sgRNA as both sequences (degraded).")

    logger.info(f"Columns: seq={seq_col}, target={target_col}, on={on_col}, off={off_col}")

    return train_df, val_df, seq_col, target_col, on_col, off_col

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer_encoder, optimizer_heads, device, phase=1):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    on_target_loss_sum = 0.0
    off_target_loss_sum = 0.0
    batch_count = 0
    
    for batch_idx, (onehot, labels, on_t, off_t, sgrna_strs, target_strs) in enumerate(train_loader):
        onehot = onehot.to(device)
        labels = labels.to(device)
        on_t = on_t.to(device)
        off_t = off_t.to(device)

        on_pred, off_pred = model(
            onehot, labels, task_type='both',
            sgrna_strs=sgrna_strs, target_strs=target_strs,
        )
        
        # Compute loss
        on_t_clamped = torch.clamp(on_t, 0.0, 1.0)
        loss_on = nn.L1Loss()(on_pred, on_t_clamped)
        loss_off = nn.BCELoss()(off_pred, torch.clamp(off_t, 0.0, 1.0))
        
        # Phase-specific weighting
        if phase == 1:
            # Phase 1: Joint training
            loss = 5.0 * loss_on + 0.5 * loss_off
        else:
            # Phase 2: On-target focused
            loss = 10.0 * loss_on + 0.5 * loss_off
        
        # Backward pass
        optimizer_encoder.zero_grad()
        optimizer_heads.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer_encoder.step()
        optimizer_heads.step()
        
        # Accumulate losses
        total_loss += loss.item()
        on_target_loss_sum += loss_on.item()
        off_target_loss_sum += loss_off.item()
        batch_count += 1
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"Batch {batch_idx+1}: Loss={loss.item():.4f}, On={loss_on.item():.4f}, Off={loss_off.item():.4f}")
    
    return {
        'total_loss': total_loss / batch_count,
        'on_target_loss': on_target_loss_sum / batch_count,
        'off_target_loss': off_target_loss_sum / batch_count
    }

def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    on_target_loss_sum = 0.0
    off_target_loss_sum = 0.0
    batch_count = 0
    
    all_on_pred = []
    all_on_true = []
    all_off_pred = []
    all_off_true = []
    
    with torch.no_grad():
        for onehot, labels, on_t, off_t, sgrna_strs, target_strs in val_loader:
            onehot = onehot.to(device)
            labels = labels.to(device)
            on_t = on_t.to(device)
            off_t = off_t.to(device)

            on_pred, off_pred = model(
                onehot, labels, task_type='both',
                sgrna_strs=sgrna_strs, target_strs=target_strs,
            )
            
            on_t_clamped = torch.clamp(on_t, 0.0, 1.0)
            loss_on = nn.L1Loss()(on_pred, on_t_clamped)
            loss_off = nn.BCELoss()(off_pred, torch.clamp(off_t, 0.0, 1.0))
            loss = 5.0 * loss_on + 0.5 * loss_off
            
            total_loss += loss.item()
            on_target_loss_sum += loss_on.item()
            off_target_loss_sum += loss_off.item()
            batch_count += 1
            
            all_on_pred.extend(on_pred.cpu().numpy().flatten())
            all_on_true.extend(on_t.cpu().numpy().flatten())
            all_off_pred.extend(off_pred.cpu().numpy().flatten())
            all_off_true.extend(off_t.cpu().numpy().flatten())
    
    # Calculate metrics
    from scipy.stats import spearmanr
    on_mask = np.array(all_on_true) > 0
    if on_mask.sum() > 0:
        on_spearman = spearmanr(np.array(all_on_true)[on_mask], np.array(all_on_pred)[on_mask])[0]
    else:
        on_spearman = 0.0
    
    from sklearn.metrics import roc_auc_score
    off_mask = np.array(all_off_true) >= 0
    if off_mask.sum() > 0:
        off_auroc = roc_auc_score(np.array(all_off_true)[off_mask] > 0.5, np.array(all_off_pred)[off_mask])
    else:
        off_auroc = 0.0
    
    return {
        'total_loss': total_loss / batch_count,
        'on_target_loss': on_target_loss_sum / batch_count,
        'off_target_loss': off_target_loss_sum / batch_count,
        'on_target_spearman': on_spearman,
        'off_target_auroc': off_auroc
    }

# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=" * 80)
    logger.info("TWO-PHASE ON-TARGET FOCUSED TRAINING")
    logger.info("=" * 80)
    
    # Load data
    train_df, val_df, seq_col, target_col, on_col, off_col = load_data()

    # Create datasets
    encoder = SequenceEncoder(device='cpu')

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, df, seq_col, target_col, on_col, off_col, encoder, seq_len=23):
            self.df = df.reset_index(drop=True)
            self.seq_col = seq_col
            self.target_col = target_col
            self.on_col = on_col
            self.off_col = off_col
            self.encoder = encoder
            self.seq_len = seq_len

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            seq = str(row[self.seq_col])
            if self.target_col is not None:
                tgt = str(row[self.target_col])
            else:
                tgt = seq
            max_len = self.seq_len
            seq = seq.ljust(max_len, 'A')[:max_len]
            tgt = tgt.ljust(max_len, 'A')[:max_len]

            on_val = float(row[self.on_col]) if self.on_col and pd.notna(row[self.on_col]) else 0.0
            off_val = float(row[self.off_col]) if self.off_col and pd.notna(row[self.off_col]) else 0.0

            onehot = self.encoder.one_hot_encode(seq)
            label = self.encoder.label_encode(seq, add_start_token=False)

            return onehot, label, on_val, off_val, seq, tgt

    def collate_fn(batch):
        onehots, labels, ons, offs, sgrna_strs, target_strs = zip(*batch)
        
        # Pad onehots
        max_len = max(x.shape[0] for x in onehots)
        onehot_padded = []
        for x in onehots:
            if x.shape[0] < max_len:
                pad = torch.zeros(max_len - x.shape[0], 4)
                x = torch.cat([x, pad], dim=0)
            onehot_padded.append(x)
        onehots = torch.stack(onehot_padded)
        
        # Pad labels
        max_len = max(x.shape[0] for x in labels)
        label_padded = []
        for x in labels:
            if x.shape[0] < max_len:
                pad = torch.zeros(max_len - x.shape[0], dtype=torch.long)
                x = torch.cat([x, pad], dim=0)
            label_padded.append(x)
        labels = torch.stack(label_padded)
        
        ons = torch.tensor(ons, dtype=torch.float32).unsqueeze(1)
        offs = torch.tensor(offs, dtype=torch.float32).unsqueeze(1)

        return onehots, labels, ons, offs, list(sgrna_strs), list(target_strs)

    train_ds = SimpleDataset(train_df, seq_col, target_col, on_col, off_col, encoder)
    val_ds = SimpleDataset(val_df, seq_col, target_col, on_col, off_col, encoder)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = CRISPRUniPredict(seq_len=23, device=str(DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Setup optimizers
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'head' in name or 'fusion' in name:
            head_params.append(param)
        else:
            encoder_params.append(param)
    
    optimizer_encoder = torch.optim.AdamW(encoder_params, lr=1e-3, weight_decay=0.01)
    optimizer_heads = torch.optim.AdamW(head_params, lr=2e-3, weight_decay=0.01)
    
    logger.info(f"Encoder params: {len(encoder_params)}")
    logger.info(f"Head params: {len(head_params)}")
    
    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: JOINT TRAINING (Epochs 1-30)")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    history = {
        'train_losses': [],
        'val_losses': [],
        'on_target_spearman': [],
        'off_target_auroc': []
    }
    
    # Phase 1: Joint training
    for epoch in range(1, 31):
        logger.info(f"\nEpoch {epoch}/30 (Phase 1)")
        
        train_metrics = train_epoch(model, train_loader, None, optimizer_encoder, optimizer_heads, DEVICE, phase=1)
        val_metrics = validate(model, val_loader, DEVICE)
        
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
        logger.info(f"On-target Spearman: {val_metrics['on_target_spearman']:.4f}")
        logger.info(f"Off-target AUROC: {val_metrics['off_target_auroc']:.4f}")
        
        history['train_losses'].append(train_metrics['total_loss'])
        history['val_losses'].append(val_metrics['total_loss'])
        history['on_target_spearman'].append(val_metrics['on_target_spearman'])
        history['off_target_auroc'].append(val_metrics['off_target_auroc'])
        
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(model.state_dict(), 'models/checkpoints/best_on_target_focused.pt')
            logger.info("✓ Best model saved")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: ON-TARGET FOCUSED TRAINING (Epochs 31-60)")
    logger.info("=" * 80)
    
    # Phase 2: On-target focused
    for epoch in range(31, 61):
        logger.info(f"\nEpoch {epoch}/60 (Phase 2)")
        
        train_metrics = train_epoch(model, train_loader, None, optimizer_encoder, optimizer_heads, DEVICE, phase=2)
        val_metrics = validate(model, val_loader, DEVICE)
        
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
        logger.info(f"On-target Spearman: {val_metrics['on_target_spearman']:.4f}")
        logger.info(f"Off-target AUROC: {val_metrics['off_target_auroc']:.4f}")
        
        history['train_losses'].append(train_metrics['total_loss'])
        history['val_losses'].append(val_metrics['total_loss'])
        history['on_target_spearman'].append(val_metrics['on_target_spearman'])
        history['off_target_auroc'].append(val_metrics['off_target_auroc'])
        
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(model.state_dict(), 'models/checkpoints/best_on_target_focused.pt')
            logger.info("✓ Best model saved")
    
    # Save history
    with open('training_history_on_target_focused.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best on-target Spearman: {max(history['on_target_spearman']):.4f}")
    logger.info(f"Best off-target AUROC: {max(history['off_target_auroc']):.4f}")
    logger.info(f"Best model saved to: models/checkpoints/best_on_target_focused.pt")

if __name__ == '__main__':
    main()
