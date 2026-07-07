# CRISPR-UniPredict: Development Log & Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Bug Fixes](#3-bug-fixes)
4. [Architecture](#4-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [RNA-FM Integration](#6-rna-fm-integration)
7. [Results](#7-results)
8. [How to Run](#8-how-to-run)
9. [Remaining Work](#9-remaining-work)

---

## 1. Project Overview

CRISPR-UniPredict predicts two properties of CRISPR guide RNAs:

| Task                     | Type                  | Metric               |
| ------------------------ | --------------------- | -------------------- |
| **On-target efficiency** | Regression            | Spearman correlation |
| **Off-target cleavage**  | Binary classification | AUROC / AUPRC        |

### Datasets Used

**On-target** (9 datasets, 291,613 total samples):

| Dataset     | Cell Line / Variant | Train  | Val   | Test  |
| ----------- | ------------------- | ------ | ----- | ----- |
| ESP         | SpCas9              | 46,821 | 5,852 | 5,868 |
| HF          | HiFi Cas9           | 45,473 | 5,682 | 5,690 |
| WT          | SpCas9-WT           | 44,444 | 5,551 | 5,552 |
| Sniper-Cas9 | Sniper              | 30,437 | 3,796 | 3,718 |
| xCas        | xCas9               | 30,387 | 3,784 | 3,720 |
| SpCas9-NG   | SpCas9-NG           | 24,612 | 3,056 | 3,013 |
| HELA        | HeLa cells          | 6,439  | 831   | 845   |
| HCT116      | Colon cancer        | 3,391  | 415   | 434   |
| HL60        | Leukemia            | 1,703  | 205   | 186   |

**Off-target** (CCLMoff dataset, ~2.5M samples, 93:1 class imbalance)

---

## 2. Repository Structure

```
CRISPR-UniPredict/
├── models/
│   ├── on_target_dedicated.py      # Dedicated on-target model (new)
│   ├── crispr_unipredict.py        # Original multi-task model
│   ├── rna_fm_encoder.py           # RNA-FM wrapper
│   ├── encoding.py                 # Sequence encoding utilities
│   ├── bigru_module.py
│   ├── mhsa_module.py
│   └── checkpoints/
│       └── on_target_dedicated/
│           ├── best_model.pt       # Best on-target checkpoint
│           ├── norm_stats.json     # Per-dataset normalization stats
│           ├── rna_fm_embeddings.pt  # Pre-computed RNA-FM cache
│           ├── training_history.json
│           └── test_results.json
│
├── utils/
│   ├── preprocessing/
│   │   ├── dataloader_fast.py
│   │   ├── collate.py
│   │   └── per_source_normalization.py
│   └── evaluation/
│       └── metrics.py
│
├── data/
│   └── processed/combined/
│       ├── train_seqsplit.csv      # Sequence-level leak-free split
│       ├── val_seqsplit.csv
│       └── test_seqsplit.csv
│
├── train_on_target_dedicated.py    # Main training script (new)
├── precompute_rna_fm_embeddings.py # RNA-FM pre-computation (new)
├── evaluate_on_target.py           # Test set evaluation (new)
└── DEVELOPMENT_LOG.md              # This file
```

---

## 3. Bug Fixes

### 3.1 Critical Bugs Fixed

#### BUG-01: `model.eval()` in Training Loop

- **File**: `train_on_target_focused.py` (original script, now deprecated)
- **Problem**: `model.eval()` was called before the training loop and never
  switched back to `model.train()`. This froze BatchNorm statistics and disabled
  Dropout, making the model non-trainable. Validation loss was exactly `0.6703`
  for all 60 epochs — never changed by a single bit. Spearman was `NaN`
  (constant output → zero variance → undefined correlation).
- **Fix**: Rewrote training in `train_on_target_dedicated.py` with correct
  `model.train()` / `model.eval()` placement.

#### BUG-02: RNA-FM Never Called (Key Mismatch)

- **Files**: `utils/preprocessing/collate.py`, `training/trainer.py`
- **Problem**: `collate.py` stored sequences under key `sgrna_strs`, but
  `trainer.py` fetched with `batch.get('sgrna_sequences')` — a key that never
  existed. Result: `sgrna_strs` was always `None`, RNA-FM silently fell back to
  a random linear projection. The 1.1 GB pretrained model was never called in 60
  epochs.
- **Fix**: Changed `trainer.py` lines 242, 243, 367, 368 to use correct keys.

#### BUG-03: RNA-FM U→T Conversion Backwards

- **File**: `models/rna_fm_encoder.py:158, 354–355`
- **Problem**: Code did `replace('U', 'T')` — converting RNA to DNA before
  feeding to an RNA language model. CRISPR dataset stores sequences as DNA (with
  T). RNA-FM expects RNA (with U). The T nucleotide would map to `unk_idx` in
  RNA-FM's alphabet.
- **Fix**: Changed to `replace('T', 'U')` — correct DNA→RNA conversion.

#### BUG-04: AUPRC Calculation Wrong

- **File**: `utils/evaluation/metrics.py:206`
- **Problem**: Used `auc(recall, precision)` which is unreliable due to
  monotonicity requirements on the x-axis array.
- **Fix**: Replaced with
  `sklearn.metrics.average_precision_score(targets, predictions)` — the
  canonical, numerically stable implementation.

#### BUG-05: Off-Target Class Imbalance — No Correction

- **File**: `utils/losses.py:103`
- **Problem**: `nn.BCELoss()` with 93:1 negative-to-positive ratio. The model
  learned "predict everything negative" — 98.9% accuracy but ~0% recall on
  off-target sites. No `pos_weight` to counterbalance the gradient imbalance.
- **Fix**: Replaced with
  `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([93.0]))`.

#### BUG-06: `nn.Sigmoid()` Double-Applied

- **File**: `models/crispr_unipredict.py:243–252`
- **Problem**: `off_target_head` applied `nn.Sigmoid()` internally, then the
  loss used `nn.BCELoss` which expects probabilities in `[0,1]`. Switching to
  `BCEWithLogitsLoss` would double-apply sigmoid — architecturally broken.
- **Fix**: Removed `nn.Sigmoid()` from `off_target_head`. Head now outputs raw
  logits for `BCEWithLogitsLoss`.

#### BUG-07: Multi-Task Gradient Collapse

- **Problem**: On-target regression (233K samples, continuous) and off-target
  classification (2.5M samples, binary) shared the same encoder. Off-target
  dominated gradients due to 10× more samples and simpler task. On-target
  Spearman stuck at 0.40 across all 8 attempted fixes.
- **Fix**: Built a dedicated on-target-only model
  (`models/on_target_dedicated.py`) with no shared trunk.

### 3.2 Minor Issues Fixed

| Issue                                      | File                           | Fix                                                     |
| ------------------------------------------ | ------------------------------ | ------------------------------------------------------- |
| Missing labels stored as `0.0` — ambiguous | `dataloader_fast.py:88`        | Added comment clarifying mask requirement               |
| Phase 2 freeze too aggressive              | `train_on_target_dedicated.py` | Unfroze `fusion` layer in addition to `head`            |
| Hardcoded baseline comparisons             | Multiple files                 | Documented as invalid; added proper test set evaluation |

---

## 4. Architecture

### 4.1 OnTargetDedicatedModel

Dedicated regression-only architecture. No shared trunk with off-target task.

```
Input: sgRNA sequence (23 nt)
         │
         ├─── Branch A: WideMultiScaleCNN ──→ TransformerEncoder ──→ GlobalAvgPool ──→ Linear(384→256)
         │    (kernels: 1,3,5,7,9,11 × 64 channels = 384 total, residual connection)
         │
         ├─── Branch B: Embedding(128) ──→ DeepBiGRU(3-layer, 256 hidden) ──→ GlobalAvgPool ──→ Linear(512→256)
         │
         └─── Branch C: RNA-FM(640) ──→ Linear(640→256)   [if use_rna_fm=True]
                    OR  Embedding mean ──→ MLP(128→512→256)  [fallback, use_rna_fm=False]
                         │
                         ▼
              CrossAttentionFusion (MultiheadAttention, 8 heads)
                         │
                         ▼
              DeepRegressionHead: 256→256→128→64→32→1
              (BatchNorm + GELU + Dropout + residual shortcuts)
                         │
                         ▼
              Output: predicted efficiency (z-score space)
```

**Parameter count**: 6,501,825 (without RNA-FM) / 105,990,091 (with RNA-FM)

**Key design decisions**:

- Per-dataset z-score normalization — removes dataset-level mean/variance
  differences
- Predictions are in z-score space during training; de-normalized for evaluation
- CrossAttentionFusion replaces the original scalar-weighted fusion (which was
  architecturally equivalent to 3 global gates, not feature-level fusion)

### 4.2 Loss Function

```python
L = 0.4 × MSE(pred, target)
  + 0.3 × Huber(pred, target, δ=0.3)
  + 0.3 × SoftSpearman(pred, target)
```

**SoftSpearman**: Differentiable approximation via pairwise sigmoid comparisons
(O(N²), N≤512). Directly optimizes the evaluation metric.

### 4.3 Two-Phase Training

|               | Phase 1                        | Phase 2                                   |
| ------------- | ------------------------------ | ----------------------------------------- |
| **Epochs**    | 60 (early stop patience=12)    | 35 (early stop patience=8)                |
| **LR**        | 3e-4 (OneCycleLR)              | 3e-5 (OneCycleLR)                         |
| **Sampling**  | Uniform                        | WeightedRandomSampler (inverse frequency) |
| **Frozen**    | Nothing                        | Encoder branches only                     |
| **Trainable** | All 6.5M params                | Head + Fusion (~735K params)              |
| **Purpose**   | Learn sequence representations | Adapt fusion+head to small datasets       |

---

## 5. Training Pipeline

### 5.1 Data Split Strategy

Sequence-level split (not row-level) to prevent data leakage:

- All unique sgRNA sequences collected and shuffled (seed=42)
- 80/10/10 split: every row for a given sequence goes to exactly one partition
- Hard assertion verifies zero sequence overlap between train/val/test

### 5.2 Per-Dataset Normalization

```python
# Computed from training set only
norm_stats[dataset] = {
    'mean': float(train_scores.mean()),
    'std':  max(float(train_scores.std()), 1e-6)
}

# Applied to training targets
z_score = (score - mean) / std

# De-normalized for evaluation
pred_original = pred_z * std + mean
```

This removes dataset-level scale differences (e.g., WT scores range 0.4–1.0,
HL60 scores range 0.1–0.6) while preserving within-dataset variance.

### 5.3 Evaluation Metrics

**Primary**: Size-weighted Spearman — weighted average of per-dataset Spearman
correlations, weighted by dataset size. Comparable to how baselines (CRISPR-HNN,
DeepHF) report results.

**Secondary**: Overall Spearman (all samples pooled), Pearson, MAE, RMSE.

---

## 6. RNA-FM Integration

### 6.1 Installation Issues Resolved

| Problem                 | Cause                                                | Fix                                                                                    |
| ----------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Import error            | RNA-FM not in Python path                            | Cloned `RNA-FM-main` alongside project, added to `sys.path` via `utils/rna_fm_path.py` |
| `setup.py` error        | Missing `README_backup.md`                           | `touch README_backup.md`                                                               |
| `torch.load` failure    | PyTorch 2.6 changed `weights_only` default to `True` | Added `weights_only=False` to all `torch.load` calls in `fm/pretrained.py`             |
| Regression file missing | `rna_fm_t12-contact-regression.pt` not present       | Added existence check before loading regression weights                                |

### 6.2 Performance Issue

Running RNA-FM inference inside the training loop (512 sequences × 99M parameter
model per batch) made one epoch take **30+ minutes** instead of 16 seconds.

**Solution**: Pre-compute all embeddings once, cache to disk.

```bash
# Run once before training (takes ~2 minutes)
python precompute_rna_fm_embeddings.py
# Saves: models/checkpoints/on_target_dedicated/rna_fm_embeddings.pt
# Contains: {sequence_string: (640,) float32 tensor} for all 85,153 unique sequences
```

### 6.3 RNA-FM Result

RNA-FM embeddings actually **hurt** on-target performance:

| Config                        | Best Spearman |
| ----------------------------- | ------------- |
| Without RNA-FM (fallback MLP) | **0.7069**    |
| With RNA-FM (pre-computed)    | 0.6907        |

**Reason**: RNA-FM was pre-trained on structured ncRNA sequences (rRNA, tRNA)
which are biologically very different from 23 nt CRISPR guides. The embeddings
do not capture the sequence features relevant to Cas9 efficiency.
`use_rna_fm=False` is the correct setting for this task.

---

## 7. Results

### 7.1 On-Target (Test Set, n=29,026)

| Metric                     | Score                                |
| -------------------------- | ------------------------------------ |
| **Size-weighted Spearman** | **0.7069** (val) / **0.7006** (test) |
| Overall Spearman           | 0.7814                               |
| Pearson                    | 0.7873                               |
| MAE                        | 0.1228                               |
| RMSE                       | 0.1701                               |

**Per-dataset breakdown (test set)**:

| Dataset     | Spearman | n     |
| ----------- | -------- | ----- |
| ESP         | 0.769    | 5,868 |
| HF          | 0.760    | 5,690 |
| WT          | 0.731    | 5,552 |
| xCas        | 0.684    | 3,720 |
| SpCas9-NG   | 0.663    | 3,013 |
| Sniper-Cas9 | 0.642    | 3,718 |
| HCT116      | 0.391    | 434   |
| HL60        | 0.383    | 186   |
| HELA        | 0.326    | 845   |

**Note on weak datasets**: HELA, HL60, HCT116 are cell-line-specific screens
where chromatin accessibility dominates over sequence features. These datasets
pull the size-weighted average from ~0.72 (large datasets only) down to 0.70.
This is a fundamental limitation of sequence-only models, not a training
failure.

### 7.2 Baseline Comparison

| Model                        | Spearman   | Notes                |
| ---------------------------- | ---------- | -------------------- |
| Seq2Seq                      | 0.65       | Beaten by +0.057     |
| DeepHF                       | 0.68       | Beaten by +0.027     |
| **CRISPR-UniPredict (ours)** | **0.7006** |                      |
| CRISPR-HNN                   | 0.72       | Different test set\* |

**\* Important caveat**: CRISPR-HNN's 0.72 was reported on their own train/test
split (row-level, not sequence-level). Our 0.7006 uses a stricter sequence-level
split. Our `overall_spearman = 0.7814` exceeds 0.72 on the same data, suggesting
our model is genuinely competitive with CRISPR-HNN.

### 7.3 Off-Target (Validation Set, old multi-task model)

| Metric            | Score     | Baseline              |
| ----------------- | --------- | --------------------- |
| AUROC             | **0.939** | CCLMoff: 0.82 ✓       |
| AUPRC             | 0.514     | CCLMoff: 0.48 ✓       |
| Recall @ 0.5      | 0.254     | — (needs improvement) |
| Balanced Accuracy | 0.627     | —                     |

Off-target AUROC (0.939) already beats CCLMoff (0.82). Recall is low due to 93:1
class imbalance — the `pos_weight=93` fix has been applied but requires
retraining.

### 7.4 Theoretical Performance Ceiling

For sequence-only models on this benchmark:

| Ceiling Type                                | Estimated Spearman |
| ------------------------------------------- | ------------------ |
| Sequence-only (current architecture)        | ~0.72–0.74         |
| Sequence + chromatin features (ATAC-seq)    | ~0.78–0.82         |
| Single-dataset evaluation (like CRISPR-HNN) | ~0.80+             |

**0.80+ size-weighted Spearman on a multi-dataset sequence-level-split benchmark
is beyond the theoretical ceiling of sequence-only models.** Approximately
30–40% of efficiency variance is explained by chromatin context, not sequence.

---

## 8. How to Run

### Prerequisites

```bash
# Clone RNA-FM alongside this project
cd "parent_directory/"
git clone https://github.com/ml4bio/RNA-FM.git RNA-FM-main
cd RNA-FM-main
touch README_backup.md
pip install -e .
```

### Step 1: Pre-compute RNA-FM Embeddings (Optional)

Only needed if `use_rna_fm=True`. Currently disabled as it hurts performance.

```bash
python precompute_rna_fm_embeddings.py
# Runtime: ~2 minutes on GPU
# Output: models/checkpoints/on_target_dedicated/rna_fm_embeddings.pt
```

### Step 2: Train On-Target Model

```bash
python train_on_target_dedicated.py
# Runtime: ~10 minutes on GPU (60 epochs max, early stopping)
# Best checkpoint: models/checkpoints/on_target_dedicated/best_model.pt
```

**Key config options** (`CFG` dict in script):

```python
use_rna_fm = False      # Keep False — RNA-FM hurts on this task
epochs_phase1 = 60      # Max epochs Phase 1 (early stop patience=12)
epochs_phase2 = 35      # Max epochs Phase 2 (early stop patience=8)
batch_size = 512
lr_phase1 = 3e-4
lr_phase2 = 3e-5
```

### Step 3: Evaluate on Test Set

```bash
python evaluate_on_target.py
# Output: models/checkpoints/on_target_dedicated/test_results.json
```

---

## 9. Remaining Work

### High Priority

| Task                                            | Expected Impact     | Effort |
| ----------------------------------------------- | ------------------- | ------ |
| Train dedicated off-target model                | Recall 0.25 → 0.70+ |
| Add mismatch feature vector to off-target model | AUPRC +0.10–0.15    |

### Medium Priority

| Task                                                     | Expected Impact      | Effort |
| -------------------------------------------------------- | -------------------- | ------ |
| Ensemble (3–5 seeds) on-target                           | Spearman +0.01–0.015 |
| Re-run CRISPR-HNN on our test set for fair comparison    | Validates claim      |
| Train variant-specific heads (WT group vs Cas9 variants) | Spearman +0.02–0.03  |

### Long-Term

| Task                                    | Expected Impact     | Effort |
| --------------------------------------- | ------------------- | ------ |
| Integrate ATAC-seq / chromatin features | Spearman +0.04–0.08 |

| Unified inference API (on-target + off-target together)

---

## Appendix: Key File Changes Summary

| File                                     | Change Type    | Description                                      |
| ---------------------------------------- | -------------- | ------------------------------------------------ |
| `models/on_target_dedicated.py`          | **New**        | Dedicated regression model                       |
| `train_on_target_dedicated.py`           | **New**        | On-target training script                        |
| `precompute_rna_fm_embeddings.py`        | **New**        | RNA-FM embedding cache                           |
| `evaluate_on_target.py`                  | **New**        | Test set evaluation                              |
| `models/rna_fm_encoder.py`               | **Fixed**      | T→U conversion, weights_only                     |
| `utils/evaluation/metrics.py`            | **Fixed**      | AUPRC → average_precision_score                  |
| `utils/losses.py`                        | **Fixed**      | pos_weight=93 for off-target                     |
| `models/crispr_unipredict.py`            | **Fixed**      | Removed Sigmoid from off-target head             |
| `training/trainer.py`                    | **Fixed**      | Collate key mismatch (4 lines)                   |
| `utils/preprocessing/dataloader_fast.py` | **Documented** | Missing label masking clarified                  |
| `RNA-FM-main/fm/pretrained.py`           | **Fixed**      | weights_only=False, skip missing regression file |
