"""
Evaluate the trained OnTargetDedicatedModel on the held-out test set.
Usage:
    python evaluate_on_target.py
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from utils.rna_fm_path import ensure_rna_fm_import_path
ensure_rna_fm_import_path(_ROOT)

from models.on_target_dedicated import OnTargetDedicatedModel

CKPT_DIR  = Path('models/checkpoints/on_target_dedicated')
TEST_CSV  = 'data/processed/combined/test_seqsplit.csv'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NMAP = {'A': 2, 'C': 3, 'G': 4, 'T': 5, 'U': 5}
OH   = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'U': [0,0,0,1]}

def encode(seq):
    seq = seq.upper()
    oh  = torch.tensor([OH.get(c, [0,0,0,0]) for c in seq], dtype=torch.float32)
    lbl = torch.tensor([NMAP.get(c, 1) for c in seq], dtype=torch.long)
    return oh, lbl


def main():
    print("=" * 60)
    print("ON-TARGET MODEL — TEST SET EVALUATION")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(CKPT_DIR / 'best_model.pt', map_location=DEVICE, weights_only=False)
    cfg  = ckpt['cfg']
    norm_stats = ckpt['norm_stats']

    model = OnTargetDedicatedModel(
        branch_channels        = cfg['branch_channels'],
        bigru_hidden           = cfg['bigru_hidden'],
        embed_dim              = cfg['embed_dim'],
        fusion_dim             = cfg['fusion_dim'],
        dropout                = cfg['dropout'],
        num_transformer_layers = cfg['num_transformer_layers'],
        num_attention_heads    = cfg['num_attention_heads'],
        device                 = str(DEVICE),
        use_rna_fm             = cfg.get('use_rna_fm', False),
    ).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, phase {ckpt['phase']})")

    # Load test data
    df = pd.read_csv(TEST_CSV)
    df.columns = [c.lower() for c in df.columns]
    df = df[(df['task_type'] == 'on_target') & df['on_target_score'].notna()].reset_index(drop=True)
    print(f"Test on-target samples: {len(df):,}")

    # Run inference
    all_pred, all_raw, all_src = [], [], []
    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            rows = df.iloc[i:i+batch_size]
            ohs, lbls = [], []
            for _, row in rows.iterrows():
                oh, lbl = encode(row['sgrna_sequence'])
                ohs.append(oh)
                lbls.append(lbl)

            # Pad to same length
            max_len = max(o.shape[0] for o in ohs)
            oh_pad  = torch.zeros(len(ohs), max_len, 4)
            lbl_pad = torch.zeros(len(ohs), max_len, dtype=torch.long)
            for j, (o, l) in enumerate(zip(ohs, lbls)):
                oh_pad[j,  :o.shape[0]] = o
                lbl_pad[j, :l.shape[0]] = l

            pred_z = model(oh_pad.to(DEVICE), lbl_pad.to(DEVICE)).squeeze(-1).cpu().numpy()

            srcs = rows['dataset_source'].values
            raws = rows['on_target_score'].values

            # De-normalize
            pred_orig = np.empty_like(pred_z)
            for j, src in enumerate(srcs):
                if src in norm_stats:
                    pred_orig[j] = pred_z[j] * norm_stats[src]['std'] + norm_stats[src]['mean']
                else:
                    pred_orig[j] = pred_z[j]

            all_pred.extend(pred_orig.tolist())
            all_raw.extend(raws.tolist())
            all_src.extend(srcs.tolist())

    pred = np.array(all_pred)
    raw  = np.array(all_raw)
    srcs = np.array(all_src)

    # Overall metrics
    scc_overall, _ = spearmanr(pred, raw)
    pcc_overall, _ = pearsonr(pred, raw)
    mae_overall    = mean_absolute_error(raw, pred)
    rmse_overall   = float(np.sqrt(np.mean((pred - raw) ** 2)))

    # Per-dataset metrics
    per_ds = {}
    for src in np.unique(srcs):
        mask = srcs == src
        if mask.sum() < 4:
            continue
        scc, _ = spearmanr(pred[mask], raw[mask])
        per_ds[src] = {'spearman': round(float(scc), 4), 'n': int(mask.sum())}

    # Size-weighted Spearman
    total_n = sum(v['n'] for v in per_ds.values())
    size_wt_scc = sum(v['spearman'] * v['n'] for v in per_ds.values()) / total_n

    print()
    print("─" * 60)
    print("OVERALL METRICS")
    print("─" * 60)
    print(f"  Size-weighted Spearman : {size_wt_scc:.4f}  ← primary metric")
    print(f"  Overall Spearman       : {scc_overall:.4f}")
    print(f"  Pearson                : {pcc_overall:.4f}")
    print(f"  MAE                    : {mae_overall:.4f}")
    print(f"  RMSE                   : {rmse_overall:.4f}")
    print()
    print("─" * 60)
    print("PER-DATASET SPEARMAN")
    print("─" * 60)
    for src, v in sorted(per_ds.items()):
        bar = "█" * int(v['spearman'] * 20)
        print(f"  {src:20s} {v['spearman']:.4f}  {bar}  (n={v['n']:,})")
    print()
    print("─" * 60)
    print("BASELINE COMPARISON (size-weighted Spearman)")
    print("─" * 60)
    baselines = {'CRISPR-HNN': 0.72, 'DeepHF': 0.68, 'Seq2Seq': 0.65}
    for name, val in baselines.items():
        diff   = size_wt_scc - val
        status = "✓ BEATS" if diff > 0 else "✗ below"
        print(f"  {name:15s}  {val:.2f}   ours {size_wt_scc:.4f}  {status} by {abs(diff):.4f}")

    # Save results
    results = {
        'size_weighted_spearman': round(size_wt_scc, 4),
        'overall_spearman'      : round(float(scc_overall), 4),
        'pearson'               : round(float(pcc_overall), 4),
        'mae'                   : round(float(mae_overall), 4),
        'rmse'                  : round(float(rmse_overall), 4),
        'per_dataset'           : per_ds,
        'n_test_samples'        : len(df),
    }
    out = CKPT_DIR / 'test_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == '__main__':
    main()
