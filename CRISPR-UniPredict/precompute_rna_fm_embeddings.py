"""
Pre-compute RNA-FM embeddings for all unique sgRNA sequences.
Run once before training:
    python precompute_rna_fm_embeddings.py

Saves: models/checkpoints/on_target_dedicated/rna_fm_embeddings.pt
  -> dict mapping sequence string -> (640,) float32 tensor
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from utils.rna_fm_path import ensure_rna_fm_import_path
ensure_rna_fm_import_path(_ROOT)

from models.rna_fm_encoder import RNAFMEncoder

TRAIN_CSV   = 'data/processed/combined/train_seqsplit.csv'
VAL_CSV     = 'data/processed/combined/val_seqsplit.csv'
RNA_FM_PATH = 'models/pretrained/rna_fm_t12.pt'
OUT_PATH    = 'models/checkpoints/on_target_dedicated/rna_fm_embeddings.pt'
BATCH_SIZE  = 256
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print(f"Device: {DEVICE}")

    # Collect unique on-target sequences
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    train_df.columns = [c.lower() for c in train_df.columns]
    val_df.columns   = [c.lower() for c in val_df.columns]

    seqs = set()
    for df in [train_df, val_df]:
        ot = df[df['task_type'] == 'on_target']
        seqs.update(ot['sgrna_sequence'].dropna().str.upper().tolist())

    seqs = sorted(seqs)
    print(f"Unique sequences: {len(seqs):,}")

    # Load RNA-FM (fully frozen)
    print("Loading RNA-FM...")
    encoder = RNAFMEncoder(
        model_path=RNA_FM_PATH,
        freeze_layers=True,
        num_unfreeze_layers=0,
        device=DEVICE,
    )
    encoder.eval()

    # Compute embeddings in batches
    cache = {}
    print("Computing embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), BATCH_SIZE)):
            batch_seqs = seqs[i : i + BATCH_SIZE]
            embs = encoder.encode_batch_pairs(batch_seqs, batch_seqs)  # (B, 640)
            embs = embs.cpu().float()
            for seq, emb in zip(batch_seqs, embs):
                cache[seq] = emb

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"Saved {len(cache):,} embeddings → {out_path}")


if __name__ == '__main__':
    main()
