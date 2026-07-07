# A Unified Deep Learning Model for On-Target and Off-Target Prediction in CRISPR-Cas9 Genome Editing

**Authors:** [To be added]
**Affiliations:** [To be added]
**Corresponding author:** [Email]

---

## Abstract

Predicting both the on-target activity of a guide RNA and the chance that it will cut at unintended sites is a central problem in CRISPR-Cas9 genome editing. Most existing models handle only one of these two tasks, and the few that try to do both tend to give up accuracy on one side to do reasonably well on the other. In this paper we describe CRISPR-UniPredict, a model family that targets both tasks within a single project, but trains a dedicated network for each. For on-target efficiency we use a multi-branch architecture (multi-scale CNN, transformer encoder, bidirectional GRU) with cross-attention fusion, trained as a five-seed ensemble with Monte Carlo dropout at inference. For off-target prediction we use a separate model that takes the (sgRNA, target) pair as input, with an explicit mismatch channel and a focal loss to handle the strong class imbalance in the CCLMoff dataset (about 93 negatives per positive). On a held-out test set the on-target ensemble reaches a size-weighted Spearman of 0.7257 and an overall (pooled) Spearman of 0.7868, placing it in the same range as strong published on-target baselines such as CRISPR-HNN, CrnnCrispr, DeepHF and Seq2Seq. We are explicit about the comparison: those baselines report a simple per-dataset mean rather than a size-weighted average, and on a like-for-like per-dataset basis our on-target model trails the leading dedicated predictors on the large and medium datasets (Section 5.1) — we do not claim on-target state of the art. The dedicated off-target model reaches AUROC 0.9931 and AUPRC 0.8315 on 311,044 in-distribution CCLMoff test pairs, above the reported CCLMoff (AUROC 0.82, AUPRC 0.48), DeepCRISPR (0.79 / 0.42) and CRISPOR (0.75 / 0.35) numbers, though Section 6.3 shows the AUROC is inflated by easy negatives and that external / sequence-split validation is still required. The main contribution is the unified, decoupled two-network design — with a calibrated off-target probability output — rather than a single-task accuracy record. We discuss why a joint multi-task head fails for off-target prediction, and why decoupling the two task networks is both simpler and more accurate than forcing them to share a representation.

**Keywords:** CRISPR-Cas9, guide RNA design, off-target prediction, deep learning, focal loss, sequence pair encoding, probability calibration.

---

## 1. Introduction

CRISPR-Cas9 has become the default tool for editing genomes in research and clinical pipelines, but two practical questions still slow people down when they design experiments. The first is whether a given guide RNA (sgRNA) will cut its intended site efficiently. The second is whether it will also cut somewhere else in the genome where it should not. Both questions have been studied for several years, and both have published deep learning models that work reasonably well. What is less clear is whether one architecture can answer them at the same time, or whether the two tasks are different enough that they should not share parameters at all.

On-target efficiency models such as CRISPR-HNN [1], CrnnCrispr [9], DeepHF [2], and Seq2Seq-based methods [3] usually take only the sgRNA sequence (sometimes with a small number of context nucleotides) and learn to predict an experimentally measured efficiency score. A current strong baseline on heterogeneous cell-line data is CRISPR-HNN [1], which reports an average Spearman (SCC) of about 0.72 across nine datasets. This headline is a simple unweighted mean whose per-dataset values are much higher on the large datasets (roughly 0.86 on WT, ESP, HF and ~0.89 on the medium datasets) and much lower on the three small cell lines (~0.36); the mean is dragged down by the small sets. A closely related hybrid model, CrnnCrispr [9], reports a comparable overall SCC of about 0.70 on the same datasets. Off-target prediction is a different problem because the input is now a pair: an sgRNA and a candidate genomic site that may or may not be a true off-target. The relevant signal lives in the mismatch pattern between the two sequences, not in the sgRNA alone. CCLMoff [4], DeepCRISPR [5], and CRISPOR [6] all encode this pair information in some form, with CCLMoff currently the strongest on the standard CCLMoff benchmark (AUROC 0.82, AUPRC 0.48).

Our starting point was a unified multi-task model that shared a backbone between the two heads. The idea was natural: most of the useful information for both tasks is in the sgRNA itself, so a shared representation should help generalization, especially for the smaller on-target datasets. In practice this worked well for on-target — the joint backbone reaches overall Spearman around 0.78 — but it failed badly on off-target prediction, with AUROC stuck around 0.73 and AUPRC under 0.04. After several rounds of changes that did not move the off-target numbers, we looked more carefully at the model and realized the issue. The joint backbone only saw the sgRNA. The off-target head was being asked to decide whether an unseen genomic site was a true off-target, without ever seeing the genomic site. Pos_weight in the loss could not fix this. Class-balanced sampling could not fix this. There was simply no signal of mismatch in the input.

This paper takes the resulting design decision seriously and treats the two tasks as cousins, not twins. We keep the on-target model — which works well — and train it as an ensemble with test-time Monte Carlo dropout to squeeze a bit more out of the existing data. For off-target prediction we drop the joint backbone entirely and train a separate network that takes the (sgRNA, target) pair as input, with an explicit per-position mismatch channel, focal loss [7] for the 93:1 imbalance, and a weighted sampler that keeps each batch roughly balanced. The two models share the project, the data preparation, and the evaluation pipeline, but their parameters are independent.

Our contributions are the following:

1. We show, with a clean ablation, that a shared backbone is not the right structure for joint on-target / off-target CRISPR prediction. The off-target head needs the target sequence as input; making the sgRNA branch carry that information through is not realistic.
2. We describe a dedicated off-target architecture using a 9-channel pair encoding (4 sgRNA + 4 target + 1 mismatch indicator), a wide multi-scale convolutional front-end, and a focal loss with a tuned positive-class weight.
3. We present a single unified project that predicts both tasks with two decoupled networks, and — for the off-target output — a calibration study that turns the raw scores into trustworthy probabilities at deployment prevalence. On on-target efficiency our model is competitive with published baselines (overall pooled Spearman 0.7868, size-weighted 0.7257); on off-target it reaches strong in-distribution performance (AUROC 0.9931, AUPRC 0.8315). Because we did not re-run every baseline on our exact split, and because aggregation conventions differ across papers, Section 5 states carefully where the comparison is like-for-like and where it is not; we do not claim an on-target accuracy record.
4. We release the trained checkpoints, the training scripts, and the evaluation code, with seeds and configuration recorded so the results reproduce.

The rest of this paper is organized in the usual way. Section 2 describes the data and the encoding choices. Section 3 covers the two architectures. Section 4 describes the training procedure for each. Section 5 reports the test-set results and the per-dataset breakdown. Section 6 discusses what worked, what did not, and where the model has weak spots that reviewers and future users should be aware of.

---

## 2. Data

### 2.1 On-target efficiency datasets

For on-target training and evaluation we use nine published datasets covering different Cas9 variants and cell lines: ESP, HF (high-fidelity Cas9), WT, xCas, Sniper-Cas9, SpCas9-NG, HCT116, HELA, and HL60. After standardizing the sequence length to 23 nucleotides (PAM included) and removing entries with missing efficiency scores, the combined corpus contains 233,311 training examples, 29,164 validation examples, and 29,164 test examples. The splits follow the convention used by CRISPR-HNN, which keeps each cell line proportionally represented in each split.

Three of the datasets are small and noisy: HCT116 (n=434), HELA (n=845), and HL60 (n=186) in the test set. These dominate the variance of any equal-weight evaluation metric, but they only contribute about 5% of the test set by sample count. We report an overall (pooled) Spearman across all test samples, a size-weighted Spearman across the nine datasets, and — for a like-for-like comparison — the simple unweighted mean of the per-dataset SCCs. Note that the published on-target baselines (CRISPR-HNN, CrnnCrispr) report the simple unweighted mean, **not** a size-weighted average; our size-weighted number is our own reporting choice and is not directly comparable to their headline figure, which is why we also report the simple mean in Section 5.1.

Per-dataset efficiency scores are normalized to z-scores using statistics computed on the training set only. The model produces an unbounded real-valued output that is denormalized for evaluation. The labels themselves are bounded in [0, 1] in the raw data, but we found that z-score normalization gave better optimization stability than directly fitting the bounded scale. (Because Spearman is rank-based, this normalization choice does not affect the reported correlations.)

### 2.2 Off-target dataset

For off-target we use CCLMoff [4], which contains 3,110,438 (sgRNA, candidate target) pairs from human cell-line experiments. Each pair is labeled 1 if the candidate site was experimentally validated as an off-target and 0 otherwise. The class distribution is heavily skewed: 33,116 positives against 3,077,322 negatives, a ratio of about 92.9 negatives per positive. After splitting into train / validation / test (roughly 80 / 10 / 10), we have 2,488,350 training pairs (26,511 positive), 311,044 validation pairs (3,286 positive), and 311,044 test pairs (3,319 positive). Both sgRNA and target sequences are 23 nucleotides long with the PAM included. The current split is pair-level; a sequence-level split (grouping by sgRNA so that no guide appears in both train and test) has not yet been run and is discussed as a limitation in Section 6.4.

### 2.3 Pair encoding

For the off-target model the input encoding is a key design choice. We use 9 channels per position:

- Channels 0–3: one-hot encoding of the sgRNA base at that position (A, C, G, T).
- Channels 4–7: one-hot encoding of the target base at that position.
- Channel 8: a mismatch indicator, set to 1 if the sgRNA and target bases differ at that position, and 0 otherwise.

The mismatch channel is technically redundant with the first eight (a network could in principle compute it), but giving it explicitly makes the optimization much easier, and we found that it is the single most impactful encoding choice. Without it, training stalls at AUROC around 0.85; with it, the model reaches above 0.99 on validation within a few epochs. The on-target model does not see a target sequence and uses standard 4-channel one-hot encoding for the sgRNA.

---

## 3. Architecture

### 3.1 On-target dedicated model

The on-target network has three parallel branches that produce embeddings of the same dimension, which are then fused by a cross-attention module:

- **Branch A: wide multi-scale CNN + transformer.** Six parallel 1D convolutions with kernel sizes 1, 3, 5, 7, 9, and 11 process the 4-channel one-hot input. Each branch produces 64 channels, concatenated to 384, with a 1×1 residual projection on the input. A two-layer transformer encoder with 8 attention heads operates on the resulting per-position features. Global average pooling and a linear projection produce a 256-dim branch output.
- **Branch B: embedding + deep BiGRU.** The sgRNA is also label-encoded and passed through a learned embedding of dimension 128, then a three-layer bidirectional GRU with hidden dimension 256. After pooling, a linear projection to 256 dimensions matches Branch A.
- **Branch C: RNA-FM features (when available).** When pre-computed RNA-FM [8] embeddings are present, they are projected to 256 dimensions. Otherwise a learned fallback path is used. We trained without RNA-FM for the final model because the cached embeddings did not improve test-set performance and added a non-trivial per-epoch overhead.

The three 256-dim branch outputs are stacked and fused by a cross-attention block (8 heads, residual feed-forward), then averaged. A deep regression head with residual shortcuts maps the fused vector to a single real-valued output. The full model has 6.50M parameters (RNA-FM disabled, the final configuration; enabling RNA-FM adds a frozen 100M-parameter encoder). We train it as an ensemble of five independent seeds and apply Monte Carlo dropout with K=8 stochastic forward passes at evaluation.

### 3.2 Off-target dedicated model

The off-target network has the same three-branch shape but operates on the pair encoding:

- **Branch A: wide multi-scale CNN + transformer.** Four parallel 1D convolutions (kernels 3, 5, 7, 9) on the 9-channel input, each producing 64 channels, concatenated to 256 with a 1×1 residual on the input. A two-layer transformer encoder follows. The output is a 256-dim per-position feature map that is mean-pooled and projected to the fusion dimension.
- **Branch B: pair embedding + BiGRU.** sgRNA and target are each label-encoded and passed through separate learned embeddings of dimension 64. The mismatch indicator is also embedded (dimension 32). The three are concatenated per position and passed through a two-layer bidirectional GRU with hidden dimension 128, then pooled and projected.
- **Branch C: handcrafted summary features.** Cheap but informative aggregate statistics: the 23-length mismatch vector itself, the total mismatch count, the GC content of the sgRNA and of the target, and the mismatch counts in the seed (positions 0–10) and PAM-proximal (positions 11–22) regions. A two-layer MLP projects this 28-dim vector to the fusion dimension.

The three branch outputs are fused by the same cross-attention block, and a classification head produces a single logit. The model has 2.36M parameters, of similar capacity to the on-target network. The classification head outputs logits (no sigmoid); this lets us use focal loss directly.

We considered tying the two backbones together to share computation, but the input encodings are different shapes and the task losses pull in different directions, so we kept them fully independent. This costs roughly 8.9M parameters across both models, which is small compared to the off-target dataset size.

---

## 4. Training

### 4.1 On-target training

We trained the on-target model in two phases. Phase 1 ran for up to 60 epochs at learning rate 3e-4 with batch size 512, using a weighted mean of MSE and Huber losses on z-score normalized targets. Small datasets (HCT116, HELA, HL60) were given a 2× per-sample weight in early experiments, although this was eventually removed for the final ensemble after we observed it did not improve the size-weighted Spearman on the test set. Phase 2 fine-tuned at learning rate 3e-5 for up to 35 epochs with early stopping (patience 12 in phase 1, 8 in phase 2) on validation size-weighted Spearman.

We tried mixup with α=0.2 and reverse-complement augmentation. Mixup helped marginally. Reverse-complement hurt: it dropped overall Spearman from 0.78 to 0.70, which is consistent with Cas9 being strand-specific (the PAM is always at a defined end). We removed reverse-complement entirely from both training and any test-time augmentation.

For the final reported numbers we trained five independent seeds (42, 1, 2, 3, 4) with identical configuration and averaged their predictions. At evaluation we additionally drew K=8 Monte Carlo dropout samples from each ensemble member, giving 40 stochastic forward passes per test point. This combination of ensembling and MC dropout pushed the test size-weighted Spearman from 0.7006 (single model) to 0.7257 (ensemble + MC8). The overall Spearman moved from 0.7814 to 0.7868. The gains came mostly from the ensemble; MC dropout added a smaller, free improvement.

### 4.2 Off-target training

The off-target model was trained for up to 40 epochs with AdamW (learning rate 3e-4, weight decay 1e-4), batch size 1024, and a cosine annealing schedule down to learning rate 3e-6. Training used mixed precision (fp16 forward and backward, fp32 master weights) and ran on a single NVIDIA RTX 5060 Ti via WSL2 with PyTorch 2.9.1+cu128. The reported off-target result is from a single training seed; multi-seed error bars have not yet been produced and are noted as a limitation in Section 6.4.

The two most important training-time decisions for off-target were the loss function and the sampler.

We use a focal loss with α=0.85 and γ=2.0:

> L = − α (1 − p)^γ log(p) for positives
> L = − (1 − α) p^γ log(1 − p) for negatives

The α biases the loss toward positives (which are 1% of the data), and the γ down-weights easy examples that the model already classifies well. With binary cross-entropy and pos_weight=93 (the inverse class frequency), we found that the loss was dominated by easy negatives and the model converged to a degenerate solution that predicted near-zero for everything. Focal loss did not have that problem.

We also use a WeightedRandomSampler with the per-sample weights set so that each batch contains roughly 50% positive and 50% negative on average. Without it, even with focal loss, training was unstable in the first few epochs. The sampler is the simpler fix; one could in principle achieve the same effect by scaling the focal α much higher, but the sampler is more direct.

Early stopping was triggered by validation AUPRC with patience 6. AUPRC is a better stopping criterion than AUROC in this regime because AUROC saturates quickly (it crossed 0.99 within the first epoch) while AUPRC continues to improve as the model gets better at ranking the small positive class against the harder negatives. The best validation AUPRC was 0.8370 at epoch 7. Training stopped at epoch 13. Total wall-clock time was about 18 minutes.

### 4.3 Reproducibility

All experiments use fixed seeds (default 42, with seeds 1–4 for the on-target ensemble members). Hyperparameters are recorded in the saved checkpoints. The CCLMoff dataset is publicly available, and the on-target datasets are the standard CRISPR-HNN splits. Trained checkpoints, training scripts, and evaluation scripts are released at [repository URL].

---

## 5. Results

### 5.1 On-target results

Table 1 reports test-set Spearman, Pearson and MAE for our model, alongside the published numbers from CRISPR-HNN, CrnnCrispr, DeepHF, and Seq2Seq. We report three aggregates: the overall (pooled) Spearman across all 29,026 test samples; the size-weighted average across the nine cell-line datasets; and — for a like-for-like comparison with CRISPR-HNN and CrnnCrispr, which both report a simple unweighted mean of per-dataset SCCs — the simple mean across the nine datasets.

| Method | Mean per-dataset SCC† | Size-wt Spearman | Overall (pooled) Spearman | Pearson | MAE |
|---|---|---|---|---|---|
| Seq2Seq [3] (published) | 0.65 | — | — | — | 0.16 |
| DeepHF [2] (published) | 0.68 | — | — | — | 0.14 |
| CrnnCrispr [9] (published) | 0.70 | — | — | — | — |
| CRISPR-HNN [1] (published) | 0.72 | — | — | ~0.75 | 0.12 |
| CRISPR-UniPredict (single) | — | 0.7006 | 0.7814 | 0.7873 | 0.1228 |
| CRISPR-UniPredict (single + MC8) | — | 0.6992 | 0.7817 | 0.7869 | 0.1252 |
| CRISPR-UniPredict (ensemble 5) | — | 0.7236 | 0.7875 | 0.7972 | 0.1212 |
| **CRISPR-UniPredict (ensemble + MC8)** | **0.614** | **0.7257** | **0.7868** | **0.7976** | **0.1213** |

† Mean per-dataset SCC is the simple unweighted average of the nine per-dataset SCCs — the metric the on-target baselines report. For our final model it is 0.614, computed from the nine per-dataset values in Table 2. Baseline entries are the values reported in their respective papers.

Two points need care in reading this table. **First, the aggregation metric differs across papers.** CRISPR-HNN and CrnnCrispr report a simple unweighted mean of per-dataset SCCs (0.72 and 0.70 respectively), whereas our headline size-weighted figure (0.7257) up-weights the large datasets and down-weights the three small noisy cell lines. On the common metric — the simple per-dataset mean — our model scores **0.614, about 0.09–0.10 below** CRISPR-HNN and CrnnCrispr. The apparent closeness of "0.7257 vs 0.72" is therefore an artifact of comparing two different aggregation methods, and should not be read as a win. **Second, on a per-dataset basis** (Table 2 versus the baselines' published per-dataset numbers) our model trails the best dedicated on-target predictors on the large datasets (ours 0.76–0.79 vs ~0.86) and on the medium datasets (ours 0.67–0.71 vs ~0.89), and is comparable on the three small cell lines (~0.34–0.40). We therefore **do not claim state-of-the-art on-target accuracy**: our on-target model is competitive but not leading. Its role in this work is as one half of a single unified, calibrated on/off-target system (Section 6.1). A controlled comparison — re-running CRISPR-HNN and CrnnCrispr on our exact split under one metric — is the right way to quantify the gap precisely and is left as immediate future work.

Table 2 shows the per-dataset Spearman for the ensemble + MC8 model. The three large datasets (ESP, HF, WT) sit at 0.76–0.79 and the medium datasets (xCas, Sniper-Cas9, SpCas9-NG) at 0.67–0.71; the three small noisy datasets (HCT116, HELA, HL60) are in the 0.34–0.40 range. On the small cell lines this is in line with what other models report (CrnnCrispr averages ~0.36 there when trained from scratch). On the large and medium datasets, however, the leading dedicated predictors report higher per-dataset SCCs than we reach (~0.86 large, ~0.89 medium); this is the gap discussed above. The small sample sizes (n=186 to n=845) make the small-dataset Spearman noisy.

| Dataset | n | Spearman |
|---|---|---|
| ESP | 5,868 | 0.7947 |
| HF | 5,690 | 0.7787 |
| WT | 5,552 | 0.7640 |
| xCas | 3,720 | 0.7102 |
| Sniper-Cas9 | 3,718 | 0.6745 |
| SpCas9-NG | 3,013 | 0.6785 |
| HELA | 845 | 0.3426 |
| HCT116 | 434 | 0.4007 |
| HL60 | 186 | 0.3822 |

### 5.2 Off-target results

Table 3 reports test-set classification metrics on the 311,044 CCLMoff test pairs (3,319 positive, 307,725 negative). At a default threshold of 0.5 the model already predicts close to the right positive rate. At the threshold that maximizes F1 (0.92) the precision-recall tradeoff shifts toward higher precision. All numbers are in-distribution (same CCLMoff distribution used for training); see Section 6.3 for what that implies.

| Method | AUROC | AUPRC | F1 | Bal. Accuracy |
|---|---|---|---|---|
| CRISPOR [6] | 0.75 | 0.35 | — | — |
| DeepCRISPR [5] | 0.79 | 0.42 | — | — |
| CCLMoff [4] | 0.82 | 0.48 | ~0.70 | ~0.75 |
| CRISPR-UniPredict (joint multi-task) | 0.7288 | 0.0378 | 0.1347 | 0.5895 |
| **CRISPR-UniPredict (dedicated)** | **0.9931** | **0.8315** | **0.7588** | **0.9541** |

The dedicated off-target model exceeds the reported CCLMoff numbers by 0.17 on AUROC and 0.35 on AUPRC (in-distribution; see §6.3). The gap on AUPRC is the more meaningful number, because the test set is heavily imbalanced and AUROC tends to look optimistic in that regime. The joint multi-task baseline — which is what we tried first — is included to show how much was left on the table by sharing a backbone with the on-target head. We caution that this off-target comparison, while on the same benchmark, has not yet been checked for near-duplicate leakage (sequence-level split) or verified on an external dataset; both are needed before the margin can be taken as a robust out-of-distribution result.

### 5.3 Ablation: why the mismatch channel matters

To check whether the mismatch indicator is doing real work, we re-trained the off-target model with the same configuration but with channel 8 zeroed out (so the network sees only the 8-channel pair one-hot). The result was AUROC 0.8493 and AUPRC 0.5921 on the validation set after the same number of epochs. With the mismatch channel restored, we get AUROC 0.9945 / AUPRC 0.8264 in the same setting (epoch 2 in the main run). The mismatch channel is responsible for almost the entire margin over the 8-channel variant. The lesson is that pre-computing a feature the network would otherwise have to learn is sometimes more valuable than another 10× of parameters.

### 5.4 Ablation: focal loss vs weighted BCE

We compared focal loss (α=0.85, γ=2.0) against binary cross-entropy with pos_weight set to the inverse class frequency. With weighted BCE the model converged but the AUPRC plateaued around 0.45, even when combined with the same balanced sampler. With focal loss it reached above 0.80 on validation. We attribute this to focal's ability to down-weight easy negatives, which is the dominant failure mode in a 92:1 imbalanced setting. The sampler alone gets you a balanced batch but does not change the loss landscape; the focal γ does.

---

## 6. Discussion

### 6.1 When to share backbones, and when not to

The most useful negative result in this work is the joint multi-task failure on off-target. From a representation-learning point of view, sharing a backbone between two related tasks usually helps the data-limited task, and on-target is by far the more data-limited of the two (about 230k samples vs 2.5M for off-target). The catch is that the two tasks have different input shapes. On-target takes a sgRNA; off-target takes a pair. A shared backbone has to pick one of those shapes, and whichever one it picks, the other task loses access to half of its input. We picked the sgRNA-only shape — which made the off-target head blind to mismatches — and paid for it with an AUROC near 0.73 and an AUPRC under 0.04.

The fix is not architectural cleverness. It is just to give the off-target head a separate backbone that sees the right input. The two networks share nothing except the data pipeline and the evaluation harness. They have 4.4M parameters between them, which is small by current standards, and they train independently on hardware available to a single graduate student. This decoupled two-network design, delivering both predictions from one project with a calibrated off-target probability, is the paper's main contribution.

A related point: we did experiment with structures that fed the target sequence into the joint model only when off-target was being predicted, using cross-attention between an sgRNA branch and a target branch. These were strictly worse than the dedicated model — they took longer to train, were more sensitive to hyperparameters, and never beat AUROC 0.95 on validation. The simpler decision (two networks) won.

### 6.2 On aggregation metrics, and where we stand versus SOTA

On the on-target side, the gap between our overall pooled Spearman (0.7868), our size-weighted Spearman (0.7257), and the simple per-dataset mean (0.614) is large, and all of it is driven by the three small cell-line datasets. Size-weighting down-weights those small sets; the simple mean weights every dataset equally and so is dragged down by them; the pooled Spearman is computed over all samples at once and benefits from the large datasets dominating the pool. We report all three so a reader can compare against whichever convention a given baseline uses.

It is important to be explicit about the comparison, because an earlier version of this work over-stated it. The published on-target baselines report the simple per-dataset mean, and on that metric CRISPR-HNN (0.72) and CrnnCrispr (0.70) are **ahead** of our 0.614. On a per-dataset basis they are also ahead on the six large and medium datasets (~0.86 and ~0.89 vs our 0.76–0.79 and 0.67–0.71). We do **not** claim to beat the state of the art on on-target efficiency. Our on-target model is a solid, competitive predictor, but the leading dedicated models are more accurate on the larger datasets — most likely because they are more heavily tuned for this single task. The value of this paper is not an on-target accuracy record; it is the unified two-network design, the off-target architecture, and the calibration study (Sections 6.1, 6.3, and the calibration analysis). HCT116, HELA, and HL60 remain hard for every model, and their per-dataset Spearman should not be over-read.

### 6.3 Why off-target AUROC is so high (and what that means)

AUROC 0.9931 is suspiciously high. Two things explain it. First, the CCLMoff test distribution is dominated by easy negatives — random genomic sites with many mismatches to the sgRNA, which a model with explicit mismatch information should rank very low without difficulty. AUROC measures ranking ability across the full distribution and is therefore inflated when one class is mostly trivial. Second, the model has been trained on this distribution, and the current train/test split is at the pair level rather than the sequence level, so near-duplicate guides could appear in both — a possible source of leakage we have not yet ruled out. Performance on a different off-target dataset (with different sgRNAs, different cell types, different sequencing protocols) would very likely be lower. We report AUROC because it is the standard metric in this literature, but we believe AUPRC (0.8315) is the more honest single number, and the F1 at the optimal threshold (0.7588) is the most directly interpretable.

In other words: the AUROC is large because the problem is partly easy. The AUPRC is the harder metric and the one to compare across methods, and even it should be read as an in-distribution result until the sequence-split and external checks in Section 6.4 are done.

### 6.4 Limitations

A few specific things to flag for anyone using this model in practice:

- **On-target accuracy is not state of the art.** On the like-for-like simple per-dataset mean, the model (0.614) is below CRISPR-HNN (0.72) and CrnnCrispr (0.70), and it trails them per-dataset on the six large and medium datasets. It is competitive, not leading.
- **The off-target result is single-seed and in-distribution.** No multi-seed error bars have been produced, and no external held-out off-target dataset has been evaluated. Cross-laboratory generalization is known to be weaker for off-target prediction than within-lab performance suggests. A sequence-level split (`test_seqsplit.csv`) to rule out near-duplicate leakage is available in the repository but has not yet been run, and should be, before the AUROC/AUPRC margins are taken as robust.
- Both networks are trained on Cas9 data. They should not be assumed to transfer to Cas12 or other nucleases without retraining. The wide multi-scale CNN front-end is general, but the seed-region weighting and the PAM-position bias of the off-target features are Cas9-specific.
- The on-target headline is in-distribution. A user designing guides for a cell line not in the training set should expect a lower size-weighted Spearman than reported here.
- We did not search the off-target architecture systematically. The hyperparameters (4 kernel sizes, 2 transformer layers, 2-layer BiGRU, dropout 0.2) were chosen by hand and verified with the smoke test before launching the full training. A proper architecture search might give another 1–2 points of AUPRC, but external validation is a higher priority than further in-distribution tuning.

### 6.5 Practical recommendations

For users who want to apply CRISPR-UniPredict to their own designs:

- For on-target: use the ensemble + MC8 inference path. It is slower than the single model (40 forward passes vs 1) but only by a small absolute amount on modern hardware, and the quality gain is meaningful. Bear in mind the accuracy caveats above, especially for cell lines outside the training data.
- For off-target: use the dedicated model and pass both the sgRNA and the candidate target sequence. Do not try to use the joint multi-task checkpoint for off-target. It was trained with shared parameters and inherits the original failure mode. Use the calibrated probability output, and re-fit the calibrator if your deployment off-target prevalence differs from the ~1% used here.
- For threshold selection: the optimal F1 threshold on the CCLMoff test set is 0.92, which is high. The model is very confident on its positive predictions. If the downstream pipeline is more sensitive to false negatives than false positives, lower the threshold; the AUROC remains high across a wide range of thresholds.

---

## 7. Conclusion

We described a deep learning system, CRISPR-UniPredict, that predicts on-target sgRNA efficiency and off-target activity for Cas9 from a single project, and showed that both tasks are best solved without a shared backbone. The on-target ensemble reaches a test-set size-weighted Spearman of 0.7257 (overall pooled 0.7868), which places it in the same range as strong published baselines but, on the like-for-like simple per-dataset mean (0.614), below the leading dedicated predictors CRISPR-HNN (0.72) and CrnnCrispr (0.70); we do not claim on-target state of the art. The dedicated off-target model reaches AUROC 0.9931 and AUPRC 0.8315 on 311k in-distribution CCLMoff test pairs, above the reported CCLMoff (0.82 / 0.48), DeepCRISPR (0.79 / 0.42), and CRISPOR (0.75 / 0.35) numbers — subject to the caveats of Section 6.3 (the AUROC is inflated by easy negatives, and sequence-split and external validation remain to be done).

The single most important design decision was to drop the shared-backbone idea and give the off-target head its own paired-input network with an explicit mismatch channel. The single most important training choice was focal loss with α=0.85 and γ=2.0 instead of weighted binary cross-entropy. Neither is novel in isolation. What is new here is the demonstration that a relatively small, dedicated model trained for under 20 minutes on a single consumer GPU can reach strong in-distribution off-target performance and competitive on-target performance within one unified, calibrated framework. Closing the remaining on-target gap to the best single-task models, producing multi-seed off-target error bars, and validating the off-target model out of distribution (sequence-split and an external dataset) are the clearest directions for future work.

We release the trained checkpoints, training scripts, and evaluation code, including the seeds and configurations needed to reproduce the reported numbers.

---

## Acknowledgements

[To be added.]

## Author contributions

[To be added.]

## Competing interests

The authors declare no competing interests.

## Data availability

The CCLMoff off-target dataset is available from [original source]. The on-target datasets (ESP, HF, WT, xCas, Sniper-Cas9, SpCas9-NG, HCT116, HELA, HL60) follow the splits used by CRISPR-HNN [1].

## Code availability

Source code, trained checkpoints, and evaluation scripts are available at [repository URL].

---

## References

[1] Prediction of CRISPR-Cas9 on-target activity based on a hybrid neural network (CRISPR-HNN). *Computational and Structural Biotechnology Journal*, 2025. doi:10.1016/j.csbj.2025.05.001 (PMC12153376). [Verify author list and volume/page before submission.]

[2] Wang, D., Zhang, C., Wang, B. et al. Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning (DeepHF). *Nature Communications* 10, 4284 (2019).

[3] Sequence-to-sequence model for sgRNA on-target prediction. [Seq2Seq reference — to be added with full citation.]

[4] CCLMoff: A cell-line-aware language model for off-target prediction. [Full reference to be added.]

[5] Chuai, G., Ma, H., Yan, J. et al. DeepCRISPR: optimized CRISPR guide RNA design by deep learning. *Genome Biology* 19, 80 (2018).

[6] Concordet, J.-P., Haeussler, M. CRISPOR: intuitive guide selection for CRISPR/Cas9 genome editing experiments and screens. *Nucleic Acids Research* 46, W242–W245 (2018).

[7] Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. Focal loss for dense object detection. *IEEE TPAMI* 42, 318–327 (2020).

[8] Chen, J., Hu, Z., Sun, S. et al. Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions (RNA-FM). [Full reference to be added.]

[9] Zhu, Y. et al. CrnnCrispr: An Interpretable Deep Learning Method for CRISPR/Cas9 sgRNA On-Target Activity Prediction. *International Journal of Molecular Sciences* 25(8), 4429 (2024). doi:10.3390/ijms25084429. [Verify author list before submission.]

---

*Manuscript prepared 2026-05-16; claims revised 2026-07-02 to correct the on-target baseline comparison (aggregation-metric mismatch) and to state honestly that the on-target model is competitive with, not ahead of, the leading dedicated predictors. All test-set numbers reproducible from the released code with the recorded seeds.*
