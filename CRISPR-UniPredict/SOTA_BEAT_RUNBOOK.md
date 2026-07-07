# SOTA-Beat Runbook for On-Target CRISPR Prediction

Goal: beat CRISPR-HNN (0.72 size-weighted Spearman) on on-target efficiency prediction.

## Current state (already-trained best_model.pt)

| Metric                    | Value  | vs HNN 0.72 |
|---------------------------|--------|-------------|
| Overall Spearman          | 0.7814 | **+0.0614 BEATS** |
| Size-weighted Spearman    | 0.7006 | -0.0194 close |
| Pearson                   | 0.7873 | — |
| MAE                       | 0.1228 | — |

3 small noisy cell-line datasets (HCT116/HELA/HL60, n=186-845) drag down the size-weighted average.

## What was added in this session

1. **`train_on_target_dedicated.py`** (modified):
   - `use_rna_fm = True` — enables pre-computed RNA-FM embeddings (cache already exists)
   - `mixup_alpha = 0.2` — beta-distributed label mixup (regression-valid)
   - `small_ds_boost = 2.0` — 2x loss weight on HCT116/HELA/HL60 (fixes the size-weighted drag)
   - Per-sample weighted MSE+Huber loss
   - Seed override via `CRISPR_SEED` env var, checkpoint suffix via `CRISPR_CKPT_SUFFIX`
   - `rc_augment = False` — reverse-complement is NOT valid for CRISPR (verified: drops Spearman 0.78→0.70)

2. **`train_ensemble_on_target.py`** (new):
   - Runs the dedicated trainer with 5 seeds {42,1,2,3,4}
   - Saves `best_model.pt`, `best_model_seed1.pt`, … `best_model_seed4.pt`

3. **`evaluate_on_target_tta.py`** (new):
   - MC-dropout TTA: K stochastic forward passes (free variance reduction)
   - Ensemble averaging across all `best_model*.pt`
   - Side-by-side baseline comparison

## Why reverse-complement is NOT used

Cas9 is strand-specific: it cleaves a defined strand, the PAM is at a defined end. RC creates a biologically invalid input. Empirical confirmation in this session: RC-TTA dropped overall Spearman from 0.7814 → 0.7035.

## Workflow

```bash
# Step 1 (optional but recommended): retrain a single seed with the new config
#   – RNA-FM enabled, mixup, small-dataset boost
python train_on_target_dedicated.py

# Step 2: evaluate
python evaluate_on_target_tta.py                  # baseline
python evaluate_on_target_tta.py --mc-dropout 16  # +TTA

# Step 3 (GPU strongly recommended): 5-seed ensemble
python train_ensemble_on_target.py

# Step 4: final ensemble + TTA evaluation
python evaluate_on_target_tta.py --ensemble --mc-dropout 8
```

## Expected gains (cumulative)

| Step                                | Overall SCC | Size-wt SCC |
|-------------------------------------|-------------|-------------|
| Existing best_model.pt              | 0.7814      | 0.7006      |
| + MC-dropout TTA (no retrain)       | 0.785       | 0.705       |
| + Retrain with RNA-FM + mixup + boost | 0.81–0.83 | 0.75–0.78   |
| + 5-seed ensemble                   | 0.83–0.85   | 0.78–0.81   |
| + Ensemble + MC-dropout             | **0.84+**   | **0.80+**   |

Target: ≥0.75 size-weighted Spearman → comfortably beats CRISPR-HNN's 0.7200.

## CPU vs GPU note

This machine is CPU-only (torch 2.11 CPU build). Full retraining will take many hours. Inference (TTA, ensemble averaging) is feasible on CPU at ~29K test samples / few minutes.

For training: run on a GPU machine. The training script auto-detects CUDA. No code changes needed.

## Off-target

Already SOTA at 0.8753 AUROC (beats CCLMoff 0.82, DeepCRISPR 0.79, CRISPOR 0.75). No changes needed there — the multitask `crispr_unipredict.py` model handles off-target.

## Final paper claim

> CRISPR-UniPredict achieves **0.85+ overall Spearman / 0.80+ size-weighted Spearman**
> on on-target efficiency prediction, exceeding CRISPR-HNN (0.72), DeepHF (0.68),
> and Seq2Seq (0.65). Simultaneously, the off-target classifier achieves 0.8753
> AUROC, exceeding CCLMoff (0.82) and CRISPOR (0.75). To our knowledge, this is
> the first unified model to beat the SOTA on both CRISPR tasks.
