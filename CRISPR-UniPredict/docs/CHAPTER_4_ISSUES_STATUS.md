# Chapter 4 Issues Status Report — 2026-04-15

**Document date:** 2026-04-15  
**Related:** Section 4 of [FULL_SESSION_DOCUMENTATION.md](./FULL_SESSION_DOCUMENTATION.md)

---

## Executive Summary

Of the **7 issues** in Chapter 4:
- ✅ **4 are FIXED** (4.4, 4.5, 4.6, 4.7)
- ⚠️ **2 are PARTIALLY MITIGATED** (4.1, 4.2) 
- ⏳ **1 has recommendations but not implemented** (4.3)

---

## Detailed Status

### Issue 4.1: Multi-task learning and class imbalance
**Status:** ⚠️ **PARTIALLY MITIGATED**

**Problem:** ~9% on-target vs 91% off-target samples; joint training emphasizes easier task.

**Current mitigations:**
- ✅ Loss weight reweighting in `MultiTaskLoss` (file: `utils/losses.py`)
  - Configurable `on_target_weight` and `off_target_weight`
  - Default: on-target weight > off-target weight
- ✅ Balanced sampling strategy in config (file: `configs/smoke_rna_fm.yaml`)
  - `sampling.strategy: balanced`
  - `on_target_ratio: 0.5, off_target_ratio: 0.5`
- ✅ Two-phase training framework available in `train_on_target_focused.py`

**What's NOT implemented:**
- Focal loss or other advanced class imbalance techniques
- Curriculum learning strategies
- Dynamic loss weighting

**Recommendation:** The current setup is reasonable for a first pass. If on-target performance remains capped at ~0.41 Spearman after retraining, consider focal loss or curriculum learning.

---

### Issue 4.2: Task mismatch vs paper baselines
**Status:** ⚠️ **PARTIALLY MITIGATED (DESIGN ISSUE)**

**Problem:** Comparing multi-task unified model directly to single-task specialists (CRISPR-HNN, CCLMoff).

**Current approach:**
- ✅ Documented in comparison metrics (`evaluation_results.json`)
- ✅ Separate task heads for on-target and off-target
- ✅ Off-target AUROC ~0.875 (competitive vs CCLMoff ~0.82)

**What's NOT implemented:**
- Separate specialist models for on-target (would need ~0.72 Spearman to match CRISPR-HNN)
- Ensemble approach combining both branches
- Task-specific architectures

**Recommendation:** 
- **Decision:** Either accept the multi-task trade-off (0.41 on-target, 0.875 off-target) OR implement a separate on-target specialist
- **For this session:** Document clearly in Methods that unified model makes different architectural trade-offs than single-task baselines

---

### Issue 4.3: On-target data and labels
**Status:** ⏳ **DOCUMENTED BUT NOT FIXED**

**Problem:** On-target labels aggregated from multiple assays/sources with incomparable scales; caps correlation.

**Current approach:**
- ✅ Target filling from sgRNA when missing (prevents data loss)
- ✅ Multi-source aggregation workflow exists in `data/processed/` 
- Data normalization applied during preprocessing

**What's NOT implemented:**
- Per-source normalization (z-score per source before combining)
- Source-specific embedding layers
- Confidence-weighted loss per source

**Recommendation:**
If you have source metadata in the CSV, implement **per-source normalization**:
```python
# In preprocessing:
for source_id in df['source_id'].unique():
    mask = df['source_id'] == source_id
    df.loc[mask, 'on_target_score'] = (df.loc[mask, 'on_target_score'] - mean[source_id]) / std[source_id]
```

---

### Issue 4.4: DataLoader / Windows (operational)
**Status:** ✅ **FIXED**

**Problem:** `num_workers > 0` on Windows causes spawn issues; CSV load can hang.

**Fixes applied:**
- ✅ Config set `num_workers: 0` (file: `configs/smoke_rna_fm.yaml`)
- ✅ Fast dataloader implemented (file: `utils/preprocessing/dataloader_fast.py`)
- ✅ On-the-fly encoding, no large cache
- ✅ Verification: Smoke test ran successfully without hangs

**Evidence:** 
- Training log shows: `num_workers: 0`
- Dataloader loads 340K test samples in <1 minute

---

### Issue 4.5: RNA-FM not actually driving Branch C (CRITICAL CODE ISSUE)
**Status:** ✅ **FIXED**

**Problem:** `CRISPRUniPredict.forward()` did not call real RNA-FM; used fallback instead.

**Fixes applied:**
- ✅ Forward method updated to accept `sgrna_strs` and `target_strs` (file: `models/crispr_unipredict.py`)
- ✅ When strings provided, calls `self.rna_fm.encode_pair(s, t)` for real encoding
- ✅ Dataloader updated to return raw strings (file: `utils/preprocessing/dataloader_fast.py`)
- ✅ Trainer passes strings to model (file: `training/trainer.py`)

**Verification (2026-04-15 smoke test):**
- ✅ Model loaded with real RNA-FM (101.7M parameters)
- ✅ No "RNA-FM not available" fallback warning
- ✅ Training loss decreased (real encoding working)

---

### Issue 4.6: Checkpoint / hub download integrity
**Status:** ✅ **FIXED**

**Problem:** Partial checkpoint downloads (~568 MB truncated file) cause zip archive errors.

**Fixes applied:**
- ✅ Hub fallback implemented: if local load fails (OSError, RuntimeError), try hub (file: `models/rna_fm_encoder.py`)
- ✅ Verified in `_load_pretrained_model()`: tries local path first, falls back to `rna_fm_t12(None)` for hub
- ✅ Deprecated local checkpoint deleted (2026-04-12)
- ✅ Full checkpoint downloaded (~1.19 GB, valid)

**Evidence:**
- Hub load successful in comprehensive_evaluation.py
- RNA-FM parameters correctly initialized

---

### Issue 4.7: Local path `rna_fm_t12.pt` vs FM `pretrained` loader
**Status:** ✅ **FIXED**

**Problem:** Incomplete local path (missing regression sidecar) fails gracefully.

**Fixes applied:**
- ✅ Try-catch wraps local load attempt (file: `models/rna_fm_encoder.py`)
- ✅ Falls back to `rna_fm_t12(None)` for hub path if local incomplete
- ✅ No errors, transparent to user

**Evidence:**
- Code handles both complete and incomplete local paths
- Hub fallback automatic and working

---

## Summary Table

| Issue | Category | Status | Evidence |
|-------|----------|--------|----------|
| 4.1 | Class imbalance | ⚠️ Mitigated | Loss weighting, balanced sampling configured |
| 4.2 | Task mismatch | ⚠️ Mitigated | Separate heads, documented tradeoffs |
| 4.3 | Data quality | ⏳ Documented | Target filling implemented, per-source norm NOT done |
| 4.4 | Windows dataloader | ✅ FIXED | `num_workers: 0` in config |
| 4.5 | RNA-FM Branch C | ✅ FIXED | Real RNA-FM used, 101.7M params verified |
| 4.6 | Hub download | ✅ FIXED | Fallback implemented, full weight loaded |
| 4.7 | Local path fallback | ✅ FIXED | Try-catch with hub fallback working |

---

## Remaining Work (Optional Advanced Steps)

### High Priority (addresses fundamental limits)
1. **Per-source normalization** (Issue 4.3)
   - Implement if source metadata available
   - Could improve on-target by 0.01-0.05 Spearman

2. **Focal loss** (Issue 4.1)
   - If class imbalance still limiting after retraining
   - Easy to add to MultiTaskLoss

### Medium Priority (design decisions)
3. **Separate on-target specialist** (Issue 4.2)
   - If goal is to match CRISPR-HNN ~0.72 Spearman
   - Trade off: lose unified model advantage

---

## Next Validation Steps

1. ✅ **Already done (2026-04-15):**
   - Smoke test passed (4.4, 4.5, 4.6, 4.7)
   - RNA-FM verified loaded correctly

2. 🔄 **In progress:**
   - Full evaluation on 340K test set
   - Will show if 4.1/4.3 mitigations sufficient

3. ⏳ **After evaluation:**
   - If on-target ~0.41 persists → consider per-source norm or focal loss
   - If off-target ~0.875 maintained → current setup adequate for publication

---

*Last updated: 2026-04-15 19:00*
