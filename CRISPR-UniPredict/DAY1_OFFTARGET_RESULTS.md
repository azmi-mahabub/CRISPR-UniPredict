# Day 1 — Off-Target Results

Run 2026-07-02 on the local RTX 5060 Ti (Windows Python 3.10, torch 2.11+cu128).

---

## 1. Guide-level leakage check ✅ DONE — headline HOLDS

**Question:** is the off-target headline (AUROC 0.9931 / AUPRC 0.8315) inflated because
the same sgRNA appears in both train and test (guide-level leakage)? The original split
is pair-level; the existing `*_seqsplit.csv` only re-split on-target, so a guide-level
off-target split had to be built (`scripts/create_offtarget_seqsplit.py`:
2,067 unique guides, 0 guide overlap; test = 208 unseen guides / 4,503 positives).

Trained on the guide-level split, evaluated on **completely unseen guides**:

| Metric | Pair-level (headline) | Guide-level (unseen guides) | Δ |
|---|---|---|---|
| AUROC | 0.9931 | **0.9917** | −0.0014 |
| AUPRC | 0.8315 | **0.8060** | −0.0255 |
| F1 (best thr) | 0.7588 | 0.7328 | −0.0260 |
| n_test (pos) | 311,044 (3,319) | 332,419 (4,503) | — |

**Verdict:** AUPRC drops only **0.026** on guides the model never saw. This is well within
"the headline largely holds out-of-guide" — the off-target result is **not** a
guide-level leakage artifact. It generalizes to unseen guides.

Artifacts: `models/checkpoints/off_target_dedicated/best_model_seqsplit_offtarget.pt`,
`test_results_seqsplit_offtarget.json`.

Still open (not leakage): AUROC is still partly inflated by *easy negatives* (see paper
§6.3), and an **external-lab** dataset has not been evaluated (>1 week task, out of Day-1
scope). Those are separate from the guide-leakage question answered here.

---

## 2. Multi-seed error bars ⏳ RUNNING

5 seeds {42, 1, 2, 3, 4} on the pair-level split → mean ± std for AUROC / AUPRC / F1.
Replaces the single-seed number (the "no error bars" concern).

Results will be written here and to `offtarget_multiseed_summary.json` on completion.

<!-- MULTISEED_RESULTS -->

---

## 3. What changes in the paper (pending error bars)

- §5.2 / §6.3: add the guide-level leakage-check row (0.9917 / 0.8060 on 208 unseen
  guides) — evidence the headline is not leakage-driven.
- §5.2: report off-target AUPRC as **mean ± std** over 5 seeds instead of a single value.
- §6.4: update the limitation — sequence/guide-split leakage check is now DONE and passed;
  the remaining off-target to-do is external-lab validation only.
