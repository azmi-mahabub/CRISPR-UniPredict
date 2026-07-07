# Day 1 — Off-Target Rigor (multi-seed error bars + leakage check)

Goal: turn the single-seed, pair-level off-target result (AUROC 0.9931 / AUPRC 0.8315)
into a defensible one — **error bars** across seeds and a **guide-level leakage check**.

Environment: this machine's GPU (RTX 5060 Ti) works from **Windows Python 3.10**
(`torch 2.11+cu128`, `cuda True`) *and* from WSL2. Pick one; commands below assume
you run from `CRISPR-UniPredict/`.

```
# Windows:
set PY=C:\Users\USERAS\AppData\Local\Programs\Python\Python310\python.exe
# WSL2 (proven in the paper, num_workers=4 → ~18 min/run):
#   use your venv python
```

---

## What was added

| File | Purpose |
|---|---|
| `scripts/create_offtarget_seqsplit.py` | Builds the **guide-level** off-target split (train/val/test by unique sgRNA, 0 guide overlap). The existing `*_seqsplit.csv` only re-split on-target; this is the missing off-target piece. |
| `run_offtarget_multiseed.py` | Trains N seeds, evaluates each on the pair-level test set, reports **mean ± std**. |
| `run_offtarget_seqsplit.py` | Trains on the guide-level split, evaluates on unseen guides, compares to 0.9931 / 0.8315 (**the leakage check**). |
| `train_off_target_dedicated.py` | Added env overrides: `CRISPR_OFFTARGET_DATA_TRAIN/VAL/TEST`, `CRISPR_OFFTARGET_NUM_WORKERS` (defaults unchanged). |

Split already built (2,067 unique guides):
`train_offtarget_seqsplit.csv` (1,653 guides), `val` (206), `test` (208, 4,503 positives).

---

## Run order

```bash
# 1. Guide-level split — ALREADY DONE (re-run only to rebuild)
python scripts/create_offtarget_seqsplit.py

# 2. Leakage check (~18 min GPU with workers=4; single seed)
python run_offtarget_seqsplit.py

# 3. Multi-seed error bars (5 × ~18 min; idempotent — skips finished seeds)
python run_offtarget_multiseed.py
#   custom / resume:  python run_offtarget_multiseed.py --seeds 42 1 2 3 4
```

On Windows, if DataLoader workers stall, prepend `set CRISPR_OFFTARGET_NUM_WORKERS=0`
(slower per epoch but no worker-spawn issues). On WSL2 leave workers at the default 4.

---

## How to read the results

**Leakage check** (`run_offtarget_seqsplit.py`) — prints pair-level vs guide-level:
- AUPRC drop **> 0.15** → the 0.9931/0.8315 headline was materially inflated by guide-level
  leakage; report the guide-level number instead.
- drop **0.05–0.15** → some optimism; report both.
- drop **< 0.05** → the headline holds on unseen guides (best case).

**Multi-seed** (`run_offtarget_multiseed.py`) → `offtarget_multiseed_summary.json`.
Report the off-target result in the paper as **AUPRC = mean ± std (5 seeds)** instead of a
single number.

---

## Outputs

```
models/checkpoints/off_target_dedicated/
  best_model_seqsplit_offtarget.pt      test_results_seqsplit_offtarget.json   # leakage check
  best_model_seed{42,1,2,3,4}.pt        test_results_seed*.json                # multi-seed
  offtarget_multiseed_summary.json                                             # mean ± std
```

Not in scope for Day 1 (needs >1 week): external-dataset off-target validation.
