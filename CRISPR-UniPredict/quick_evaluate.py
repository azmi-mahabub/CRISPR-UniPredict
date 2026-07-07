#!/usr/bin/env python
"""Quick evaluation of trained model"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))

# Ensure RNA-FM is in path (so checkpoints with real RNA-FM can be loaded)
from utils.rna_fm_path import ensure_rna_fm_import_path
ensure_rna_fm_import_path(Path(__file__).parent)

from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder
from utils.preprocessing.dataloader_fast import FastCRISPRDataset, collate_fn
from torch.utils.data import DataLoader

def evaluate():
    print("=" * 80)
    print("EVALUATING TRAINED MODEL")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = CRISPRUniPredict(device=device)
    checkpoint = torch.load('models/checkpoints/best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded")
    
    # Load test data
    print("Loading test data...")
    test_dataset = FastCRISPRDataset('data/processed/combined/test.csv', 
                                      SequenceEncoder(device=device), 
                                      verbose=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                            num_workers=0, pin_memory=False, collate_fn=collate_fn)
    print(f"✓ Test set: {len(test_dataset)} samples")
    
    # Evaluate
    print("\nEvaluating...")
    on_target_preds = []
    on_target_true = []
    off_target_preds = []
    off_target_true = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)}")
            
            sgrna_oh = batch['sgrna_onehot'].to(device)
            sgrna_lbl = batch['sgrna_label'].to(device)
            on_target = batch['on_target_score'].to(device)
            off_target = batch['off_target_label'].to(device)
            on_target_mask = batch['on_target_mask'].to(device)
            off_target_mask = batch['off_target_mask'].to(device)
            
            on_target_pred, off_target_pred = model(sgrna_onehot=sgrna_oh, 
                                                     sgrna_label=sgrna_lbl, 
                                                     task_type='both')
            
            # Store predictions with masks
            on_target_preds.append(on_target_pred.cpu().numpy())
            on_target_true.append(on_target.cpu().numpy())
            off_target_preds.append(off_target_pred.cpu().numpy())
            off_target_true.append(off_target.cpu().numpy())
    
    # Concatenate
    on_target_preds = np.concatenate(on_target_preds).flatten()
    on_target_true = np.concatenate(on_target_true).flatten()
    off_target_preds = np.concatenate(off_target_preds).flatten()
    off_target_true = np.concatenate(off_target_true).flatten()
    
    # Compute metrics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # On-target metrics
    print("\n📊 ON-TARGET METRICS:")
    on_target_mask = on_target_true > 0
    if on_target_mask.sum() > 0:
        on_target_preds_valid = on_target_preds[on_target_mask]
        on_target_true_valid = on_target_true[on_target_mask]
        
        spearman = spearmanr(on_target_true_valid, on_target_preds_valid)[0]
        pearson = pearsonr(on_target_true_valid, on_target_preds_valid)[0]
        mae = mean_absolute_error(on_target_true_valid, on_target_preds_valid)
        rmse = np.sqrt(mean_squared_error(on_target_true_valid, on_target_preds_valid))
        
        print(f"  Spearman: {spearman:.4f} (target: 0.70-0.85)")
        print(f"  Pearson:  {pearson:.4f} (target: 0.70-0.85)")
        print(f"  MAE:      {mae:.4f} (target: <0.15)")
        print(f"  RMSE:     {rmse:.4f} (target: <0.15)")
    else:
        print("  No valid on-target samples")
    
    # Off-target metrics
    print("\n📊 OFF-TARGET METRICS:")
    off_target_mask = off_target_true >= 0
    if off_target_mask.sum() > 0:
        off_target_preds_valid = off_target_preds[off_target_mask]
        off_target_true_valid = off_target_true[off_target_mask]
        
        balanced_acc = balanced_accuracy_score(off_target_true_valid, 
                                               (off_target_preds_valid > 0.5).astype(int))
        f1 = f1_score(off_target_true_valid, (off_target_preds_valid > 0.5).astype(int))
        auroc = roc_auc_score(off_target_true_valid, off_target_preds_valid)
        auprc = average_precision_score(off_target_true_valid, off_target_preds_valid)
        
        print(f"  Balanced Accuracy: {balanced_acc:.4f} (target: >0.75)")
        print(f"  F1 Score:          {f1:.4f} (target: >0.75)")
        print(f"  AUROC:             {auroc:.4f} (target: 0.80-0.90)")
        print(f"  AUPRC:             {auprc:.4f} (target: 0.80-0.90)")
    else:
        print("  No valid off-target samples")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    evaluate()
