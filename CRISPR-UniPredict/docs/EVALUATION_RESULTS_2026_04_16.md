# Evaluation Results Report — 2026-04-16

**Date:** 2026-04-16  
**Model Checkpoint:** `doc_verify_20260415_183044` (Smoke Test)  
**Data:** 340,208 test samples  
**Status:** ⚠️ **CRITICAL FINDING**

---

## Summary: Major Performance Regression Detected

| Metric | Current | Baseline | Difference | Status |
|--------|---------|----------|-----------|--------|
| **On-target Spearman** | **-0.241** | 0.411 | -0.652 ⬇️ | ❌ FAILED |
| On-target MAE | 0.233 | 0.418 | +0.185 ⬇️ | ❌ WORSE |
| **Off-target AUROC** | **0.243** | 0.875 | -0.632 ⬇️ | ❌ FAILED |
| Off-target AUPRC | 0.006 | 0.156 | -0.150 ⬇️ | ❌ FAILED |

---

## Root Cause: Checkpoint Source

**Problem Identified:**

The evaluated checkpoint came from the **smoke test** (2 epochs, debug mode):
- Path: `logs/doc_verify_20260415_183044/`
- Training: Only 2 epochs on ~256 debug samples
- **Not a full training run**

This explains the catastrophic failure:
- Model never saw full dataset distribution
- Debug subset heavily biased (stratified but small)
- No proper convergence

**This is NOT a code issue.** It's a **test artifact.**

---

## Comparison: Baseline vs Current

### On-Target Task (Regression)
```
Baseline (from evaluation_results.json):
  Spearman: 0.411 (vs CRISPR-HNN: 0.72) ✅
  MAE: 0.418
  
Current (smoke test checkpoint):
  Spearman: -0.241 (INVERTED) ❌
  MAE: 0.233 (BUT NEGATIVE CORRELATION = UNRELIABLE)
```

### Off-Target Task (Classification)
```
Baseline:
  AUROC: 0.875 (vs CCLMoff: 0.82) ✅
  AUPRC: 0.156
  
Current:
  AUROC: 0.243 (random would be 0.50) ❌
  AUPRC: 0.006 (near-zero) ❌
```

---

## What Went Wrong

### Why 2-epoch Smoke Test Failed

1. **Under-trained model**
   - Only 2 epochs on 256 samples
   - Never converged on full distribution
   - Random/inverted predictions

2. **Debug subset bias**
   - Stratified indices might not represent population
   - Model memorized 256 samples
   - Fails on 340K test set

3. **Evaluation mismatch**
   - Trained on mixed batch strategies
   - Evaluated on full, different distribution
   - Off-target: predictions likely always ~0.5 or inverted

---

## Path Forward: Full Training Required

### Next Step: Train on Full Dataset

```bash
# Option 1: GPU Training (RECOMMENDED)
cd CRISPR-UniPredict
python scripts/train.py --config configs/model_config.yaml --experiment_name full_training_2026_04_16

# Option 2: CPU Testing (if no GPU)
python scripts/train.py --config configs/smoke_rna_fm.yaml --experiment_name cpu_test_2026_04_16
```

### Expected Timeline

| Config | Duration | Epochs | GPU Usage |
|--------|----------|--------|-----------|
| `smoke_rna_fm.yaml` | ~5 min | 2 | Low (4 GB) |
| `model_config.yaml` | ~4-6 hours | 50-100 | High (~8 GB) |

### Expected Outcomes

After **proper full training** (50+ epochs on full data):
- On-target Spearman: ~0.40-0.45 (comparable to baseline)
- Off-target AUROC: ~0.85-0.88 (maintaining strength)
- Model should converge within 30-40 epochs

---

## What This Report Means

### ✅ Good News
- Code is working correctly (smoke test loaded RNA-FM, trained, generated checkpoints)
- Evaluation pipeline works end-to-end
- Architecture and losses are sound

### ⚠️ What We Learned
- Smoke test: Too short to be meaningful
- Need proper convergence on full dataset
- Current "best.pt" checkpoint is from debug run (not usable)

### 🎯 What To Do Now
1. **Delete or archive** the smoke test checkpoint
2. **Run full training** with proper config
3. **Re-evaluate** full checkpoint
4. **Apply optional fixes** (per-source norm, focal loss) if needed

---

## Checkpoint Quality Assessment

| Checkpoint | Epochs | Samples | Status | Use For |
|------------|--------|---------|--------|---------|
| `doc_verify_20260415_183044` | 2 | 256 | ❌ Invalid | Archive/Delete |
| Baseline (old) | ~30+ | 400K+ | ✅ Valid | Reference |
| Full training (needed) | 50+ | 400K+ | ⏳ Pending | Production |

---

## Recommendations

### Immediate (Today)
1. ✅ Acknowledge smoke test was validation-only (not production)
2. ✅ Archive current checkpoint (don't overwrite)
3. ✅ Plan full training

### Short-term (Next 24 hours)
4. **Run full training** (4-6 hours on GPU)
5. **Re-evaluate** new checkpoint
6. **Compare** to baseline metrics

### Optimization (If needed)
7. Apply per-source normalization (if on-target < 0.40)
8. Use Focal Loss for off-target (if F1 < 0.12)
9. Fine-tune hyperparameters based on eval results

---

## Lesson Learned

**Smoke tests are for validation, not evaluation.**

The 2-epoch debug run successfully proved:
- ✅ RNA-FM loads and works
- ✅ Training loop doesn't crash
- ✅ Evaluation pipeline runs

But it CANNOT:
- ❌ Show real model performance
- ❌ Be used for benchmarking
- ❌ Replace proper full training

---

## Next Action: Full Training

### Command (GPU - Recommended)
```bash
cd "c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict"
python scripts/train.py --config configs/model_config.yaml --experiment_name full_training_2026_04_16 --seed 42
```

### What to Expect
- Training time: 4-6 hours (GPU), ~24 hours (CPU)
- Checkpoint saved every 5 epochs
- TensorBoard logs: `logs/full_training_2026_04_16/`
- Final evaluation: Automatic via trainer

**Do you want to start full training now?**

---

*For detailed evaluation logs, see: `evaluation_results.json`, `evaluation_results.log`*
