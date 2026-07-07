"""
Smoke test: run 1 epoch on 10k samples to verify the off-target pipeline works
end-to-end on GPU. Should finish in ~30s.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from models.off_target_dedicated import OffTargetDedicatedModel, FocalLoss
from train_off_target_dedicated import OffTargetDataset, load_split, make_sampler, evaluate


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = device == 'cuda'
    print(f'Device: {device} | AMP: {amp}')

    # ── Load small subset ──────────────────────────────────────────
    print('Loading train (sampling 10k)...')
    df_tr = load_split('data/processed/combined/train.csv')
    pos = df_tr[df_tr['off_target_label'] == 1].head(2000)
    neg = df_tr[df_tr['off_target_label'] == 0].head(8000)
    df_tr_small = pd.concat([pos, neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f'  train_small: {len(df_tr_small)}  (pos {int(df_tr_small["off_target_label"].sum())})')

    print('Loading val (sampling 5k)...')
    df_va = load_split('data/processed/combined/val.csv').head(5000)
    print(f'  val_small: {len(df_va)}  (pos {int(df_va["off_target_label"].sum())})')

    ds_tr = OffTargetDataset(df_tr_small, seq_len=23)
    ds_va = OffTargetDataset(df_va, seq_len=23)

    sampler = make_sampler(df_tr_small['off_target_label'].values, target_pos_ratio=0.5)

    dl_tr = DataLoader(ds_tr, batch_size=256, sampler=sampler, num_workers=2,
                       pin_memory=(device == 'cuda'))
    dl_va = DataLoader(ds_va, batch_size=512, shuffle=False, num_workers=2,
                       pin_memory=(device == 'cuda'))

    # ── Model ─────────────────────────────────────────────────────
    model = OffTargetDedicatedModel().to(device)
    print(f'Params: {model.count_params()["total"]:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = FocalLoss(alpha=0.85, gamma=2.0)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ── Train 1 epoch ─────────────────────────────────────────────
    print('Training 1 epoch...')
    model.train()
    t0 = time.time()
    n_batches = 0
    loss_sum = 0.0
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
            opt.step()
        loss_sum += loss.item()
        n_batches += 1

    dt = time.time() - t0
    print(f'Train epoch done: {n_batches} batches, loss={loss_sum/n_batches:.4f}, time={dt:.1f}s')

    # ── Eval ──────────────────────────────────────────────────────
    print('Evaluating...')
    metrics = evaluate(model, dl_va, device, amp)
    print('Val metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')

    print(f'\n=== SMOKE TEST PASSED ===')


if __name__ == '__main__':
    main()
