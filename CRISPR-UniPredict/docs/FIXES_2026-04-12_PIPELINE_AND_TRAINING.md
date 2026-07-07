# Pipeline & training fixes — 2026-04-12

**Last updated:** 2026-04-12  
**Related:** [FULL_SESSION_DOCUMENTATION.md](./FULL_SESSION_DOCUMENTATION.md) (broader project history)

This document records a **second engineering pass** after the initial RNA-FM wiring: what was broken, what we changed, how to verify it, and what remains optional follow-up.

---

## 1. Summary (what we fixed in this round)

| Topic | Problem | Fix |
|--------|---------|-----|
| **PYTHONPATH / RNA-FM** | Users had to set `PYTHONPATH` manually for `import fm`. | `utils/rna_fm_path.py` + call from `scripts/train.py` (and `train_on_target_focused.py`) to prepend **`RNA-FM-main`** (next to the repo) to `sys.path`. |
| **`--debug` subsets** | Debug mode took the **first N rows** of the CSV. In your data, those rows were often **off-target-only**, so logs showed **no on-target loss** / “No valid on-target samples” even when the file contains on-target labels elsewhere. | `build_stratified_debug_indices()` samples up to **256** indices per split, mixing rows with **`on_target_score`** and **`off_target_label`** present when possible. |
| **Missing `target` for on-target rows** | Many rows with **on-target scores** had **no protospacer `target`** in the CSV. The loader required valid 22–24 bp **both** sequences; **`target` NaN** failed validation, so **all on-target-labeled rows were dropped** before training. | In `FastCRISPRDataset`, **before** the validity mask: if `target` is missing but `sgrna` exists, **set `target = sgrna`** (same convention as using the guide sequence when no separate target is stored). |
| **TensorBoard + NumPy 2.x** | `Failed to initialize TensorBoard: module 'numpy' has no attribute 'bool8'`. Older TensorBoard/protobuf paths expect `np.bool8`, removed in NumPy 2. | In `Trainer._init_logging`, assign `np.bool8 = np.bool_` before importing `SummaryWriter`. |
| **`train_on_target_focused.py`** | Script did not add RNA-FM to the path; forward pass had **no** `sgrna_strs` / `target_strs`, so Branch C could not use real pair encoding. | `ensure_rna_fm_import_path()`; detect `target_sequence` / `target`; pad sequences; pass **`sgrna_strs` and `target_strs`** into `CRISPRUniPredict.forward`. |

---

## 2. Issues in detail

### 2.1 Why “no on-target samples” appeared in logs

Three separate mechanisms were involved:

1. **Debug indexing** — First-64-rows did not overlap regions of the file where `on_target_score` is non-null.  
2. **Validation split** — After the old filter, **val** could appear to have **zero** on-target rows because on-target rows lacked `target` and were removed (see below).  
3. **Trainer message** — “No valid on-target samples in batch” means the **batch mask** had no True entries for on-target; any of the above can cause that.

### 2.2 Why on-target rows disappeared from the filtered dataset

Pipeline logic (simplified):

1. Load CSV, normalize column names (`sgrna_sequence` → `sgrna`, etc.).  
2. Apply **sequence validity**: `sgrna` and `target` must be strings of length **22–24** with **ACGTU** only.  
3. If **`target` is NaN** (common for on-target efficiency-only rows), `is_valid(target)` is **false** → row is **dropped**.  
4. Result: **hundreds of thousands** of on-target labels in the raw file could become **zero** rows in `dataset.df`, which is **incorrect** for training and for debugging.

**Intended behavior:** For on-target-only records, using **sgRNA as both inputs** to a pair encoder is a **reasonable stand-in** when the dataset does not store a separate target sequence (the model still receives a consistent pair for RNA-FM and collate).

### 2.3 Stratified debug indices (after target fill)

Once missing `target` is filled from `sgrna`, the filtered training set again contains on-target labels. Stratified debug then selects roughly **half** of the debug budget from indices with non-null `on_target_score` and **half** from non-null `off_target_label` (with deduplication and fill to reach the cap). Example check after the fix: **128 + 128** in 256 sampled train indices when both label types exist in the filtered frame.

---

## 3. Files changed (this round)

| File | Role |
|------|------|
| `utils/rna_fm_path.py` | **New.** `ensure_rna_fm_import_path(project_root)` adds **`RNA-FM-main`** to `sys.path`. (A duplicate `utils/rna_fm` tree was removed 2026-04-12.) |
| `scripts/train.py` | Calls `ensure_rna_fm_import_path`; **`--debug`** uses `build_stratified_debug_indices` instead of `range(min(64, len))`. |
| `utils/preprocessing/dataloader_fast.py` | Fill missing `target` from `sgrna`; **`build_stratified_debug_indices()`** for debug sampling. |
| `training/trainer.py` | NumPy **`bool8`** compatibility for TensorBoard. |
| `train_on_target_focused.py` | RNA-FM path; lowercase columns; **target** column; collate passes **string lists** into the model. |

Earlier session (same week) already wired **Branch C** to real RNA-FM when `sgrna_strs` / `target_strs` are supplied (`crispr_unipredict.py`, collate, trainer). This round **does not duplicate** that narrative; see `FULL_SESSION_DOCUMENTATION.md` §6–7.

---

## 4. How to verify quickly

**Stratified debug + path (no manual PYTHONPATH required for `train.py`):**

```powershell
cd CRISPR-UniPredict
python scripts/train.py --config configs/smoke_rna_fm.yaml --experiment_name doc_verify --debug
```

Expect: training batches can show **non-zero on-target loss** when the filtered data contains on-target labels; TensorBoard may initialize without the `bool8` error.

**Sanity check in Python (optional):**

```python
from models.encoding import SequenceEncoder
from utils.preprocessing.dataloader_fast import FastCRISPRDataset, build_stratified_debug_indices
import pandas as pd

ds = FastCRISPRDataset("data/processed/combined/train.csv", SequenceEncoder(device="cpu"), verbose=False)
idx = build_stratified_debug_indices(ds, 256, seed=42)
df = ds.df
on_n = sum(1 for i in idx if pd.notna(df.iloc[i]["on_target_score"]))
off_n = sum(1 for i in idx if pd.notna(df.iloc[i]["off_target_label"]))
print(len(df), "filtered rows; debug idx with on:", on_n, "with off:", off_n)
```

After the **target fill**, `df["on_target_score"].notna().sum()` on the training split should be **large** (on the order of hundreds of thousands), not zero.

---

## 5. What we did *not* change (optional next steps)

- **Speed:** Branch C still runs **one RNA-FM forward per sample per batch** in a loop in the current integration. For production, **batch tokenization + single transformer forward** on GPU is the next performance step.  
- **Scientific claim:** Filling `target` with `sgrna` for missing rows is a **pragmatic encoding choice**; document it in methods if you publish.  
- **Full evaluation:** Re-run `comprehensive_evaluation.py` / paper numbers after any long retrain; this doc does not update benchmark tables.

---

## 6. Changelog entry (short)

- **2026-04-12:** Auto RNA-FM import path; stratified `--debug` indices; fill missing `target` from `sgrna` for on-target rows; TensorBoard NumPy 2 compatibility; `train_on_target_focused.py` aligned with RNA-FM string inputs.

---

*End of document.*
