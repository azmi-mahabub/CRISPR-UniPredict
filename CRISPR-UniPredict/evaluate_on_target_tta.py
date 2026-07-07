"""
Test-Time Augmentation (TTA) + Ensemble Evaluation for On-Target Model
=======================================================================
Improves Spearman over plain evaluate_on_target.py via:

  1. MC-dropout TTA: K stochastic forward passes with dropout active,
     average the predictions. Free variance reduction.
  2. Ensemble: average across multiple seeds (best_model*.pt). Pairs
     well with train_ensemble_on_target.py.

NOTE on reverse-complement: CRISPR is strand-specific (Cas9 cleaves a
defined strand, PAM is at a defined end). RC is NOT a valid CRISPR
augmentation and has been intentionally removed. Tested empirically:
RC-TTA drops Spearman 0.78 -> 0.70.

Usage:
    python evaluate_on_target_tta.py                       # plain (sanity vs evaluate_on_target.py)
    python evaluate_on_target_tta.py --mc-dropout 16       # MC-dropout TTA
    python evaluate_on_target_tta.py --ensemble            # average all best_model*.pt
    python evaluate_on_target_tta.py --ensemble --mc-dropout 8
"""

import sys
import json
import argparse
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


def pad_batch(seqs):
    ohs, lbls = [], []
    for s in seqs:
        o, l = encode(s)
        ohs.append(o); lbls.append(l)
    max_len = max(o.shape[0] for o in ohs)
    oh_pad  = torch.zeros(len(ohs), max_len, 4)
    lbl_pad = torch.zeros(len(ohs), max_len, dtype=torch.long)
    for j, (o, l) in enumerate(zip(ohs, lbls)):
        oh_pad[j,  :o.shape[0]] = o
        lbl_pad[j, :l.shape[0]] = l
    return oh_pad, lbl_pad


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt['cfg']
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
    return model, ckpt['norm_stats']


def predict(model, oh, lbl, mc_dropout_passes=0):
    if mc_dropout_passes > 0:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        with torch.no_grad():
            preds = torch.stack(
                [model(oh, lbl).squeeze(-1) for _ in range(mc_dropout_passes)],
                dim=0,
            ).mean(dim=0)
        model.eval()
    else:
        with torch.no_grad():
            preds = model(oh, lbl).squeeze(-1)
    return preds.cpu().numpy()


def denormalize(pred_z, srcs, norm_stats):
    out = np.empty_like(pred_z)
    for j, src in enumerate(srcs):
        if src in norm_stats:
            out[j] = pred_z[j] * norm_stats[src]['std'] + norm_stats[src]['mean']
        else:
            out[j] = pred_z[j]
    return out


def run_inference(models_with_stats, df, mc_dropout=0, batch_size=512, verbose=True):
    all_pred, all_raw, all_src = [], [], []
    n = len(df)

    for i in range(0, n, batch_size):
        rows = df.iloc[i:i+batch_size]
        seqs = rows['sgrna_sequence'].astype(str).str.upper().tolist()
        srcs = rows['dataset_source'].values
        raws = rows['on_target_score'].values

        oh, lbl = pad_batch(seqs)
        oh = oh.to(DEVICE); lbl = lbl.to(DEVICE)

        ensemble_preds = []
        for model, norm_stats in models_with_stats:
            pred_z = predict(model, oh, lbl, mc_dropout_passes=mc_dropout)
            ensemble_preds.append(denormalize(pred_z, srcs, norm_stats))

        batch_pred = np.mean(ensemble_preds, axis=0)
        all_pred.extend(batch_pred.tolist())
        all_raw.extend(raws.tolist())
        all_src.extend(srcs.tolist())

        if verbose and (i // batch_size) % 10 == 0:
            print(f"  batch {i//batch_size+1}/{(n+batch_size-1)//batch_size}", flush=True)

    return np.array(all_pred), np.array(all_raw), np.array(all_src)


def report(pred, raw, srcs, label, out_path=None):
    scc_overall, _ = spearmanr(pred, raw)
    pcc_overall, _ = pearsonr(pred, raw)
    mae_overall    = mean_absolute_error(raw, pred)
    rmse_overall   = float(np.sqrt(np.mean((pred - raw) ** 2)))

    per_ds = {}
    for src in np.unique(srcs):
        mask = srcs == src
        if mask.sum() < 4: continue
        scc, _ = spearmanr(pred[mask], raw[mask])
        per_ds[src] = {'spearman': round(float(scc), 4), 'n': int(mask.sum())}

    total_n = sum(v['n'] for v in per_ds.values())
    size_wt_scc = sum(v['spearman'] * v['n'] for v in per_ds.values()) / total_n

    print()
    print("=" * 60)
    print(f"RESULTS - {label}")
    print("=" * 60)
    print(f"  Size-weighted Spearman : {size_wt_scc:.4f}")
    print(f"  Overall Spearman       : {scc_overall:.4f}")
    print(f"  Pearson                : {pcc_overall:.4f}")
    print(f"  MAE                    : {mae_overall:.4f}")
    print(f"  RMSE                   : {rmse_overall:.4f}")
    print("-" * 60)
    print("Per-dataset Spearman:")
    for src, v in sorted(per_ds.items()):
        bar = "#" * int(max(0, v['spearman']) * 20)
        print(f"  {src:20s} {v['spearman']:.4f}  {bar}  (n={v['n']:,})")
    print("-" * 60)
    print("vs published baselines (size-weighted):")
    for name, val in {'CRISPR-HNN': 0.72, 'DeepHF': 0.68, 'Seq2Seq': 0.65}.items():
        diff = size_wt_scc - val
        status = "BEATS" if diff > 0 else "below"
        print(f"  {name:15s}  {val:.2f}   ours {size_wt_scc:.4f}  {status} by {abs(diff):.4f}")
    print()
    print("vs CRISPR-HNN on overall Spearman:")
    diff = scc_overall - 0.72
    status = "BEATS" if diff > 0 else "below"
    print(f"  CRISPR-HNN     0.72   ours {scc_overall:.4f}  {status} by {abs(diff):.4f}")

    if out_path:
        results = {
            'mode'                  : label,
            'size_weighted_spearman': round(size_wt_scc, 4),
            'overall_spearman'      : round(float(scc_overall), 4),
            'pearson'               : round(float(pcc_overall), 4),
            'mae'                   : round(float(mae_overall), 4),
            'rmse'                  : round(float(rmse_overall), 4),
            'per_dataset'           : per_ds,
        }
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc-dropout', type=int, default=0,  help='MC-dropout passes (0=deterministic)')
    ap.add_argument('--ensemble',   action='store_true', help='Average across all best_model*.pt')
    ap.add_argument('--ckpt',       default='best_model.pt')
    args = ap.parse_args()

    if args.ensemble:
        # Explicit ensemble members only (base seed42 + seed1..N). Do NOT glob
        # 'best_model*.pt' blindly: that also sweeps in stray checkpoints such as
        # best_model_v1_cpu.pt and silently changes the ensemble size / result.
        ckpts = sorted(list(CKPT_DIR.glob('best_model.pt')) + list(CKPT_DIR.glob('best_model_seed*.pt')))
        if not ckpts:
            raise SystemExit(f"No best_model.pt / best_model_seed*.pt in {CKPT_DIR}")
    else:
        ckpts = [CKPT_DIR / args.ckpt]

    print(f"Loading {len(ckpts)} checkpoint(s):")
    for c in ckpts: print(f"  {c.name}")

    models_with_stats = [load_model(c) for c in ckpts]

    df = pd.read_csv(TEST_CSV, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df = df[(df['task_type'] == 'on_target') & df['on_target_score'].notna()].reset_index(drop=True)
    print(f"Test samples: {len(df):,}")

    parts = []
    parts.append(f"ENSEMBLE[{len(models_with_stats)}]" if len(models_with_stats) > 1 else "SINGLE")
    if args.mc_dropout: parts.append(f"MC{args.mc_dropout}")
    label = " + ".join(parts)

    pred, raw, srcs = run_inference(
        models_with_stats, df,
        mc_dropout=args.mc_dropout,
    )

    if len(models_with_stats) == 1 and args.mc_dropout == 0:
        out_name = 'test_results_baseline.json'
    elif len(models_with_stats) == 1:
        out_name = f'test_results_mc{args.mc_dropout}.json'
    elif args.mc_dropout == 0:
        out_name = 'test_results_ensemble.json'
    else:
        out_name = f'test_results_ensemble_mc{args.mc_dropout}.json'

    report(pred, raw, srcs, label, out_path=CKPT_DIR / out_name)


if __name__ == '__main__':
    main()
