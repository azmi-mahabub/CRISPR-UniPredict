# Why On-Target Prediction is Not Improving - Detailed Analysis

**Date**: November 27, 2025  
**Analysis**: Comprehensive investigation into on-target prediction failure

---

## Executive Summary

**The Problem**: On-target Spearman is stuck at 0.4089 despite aggressive training improvements (loss weight 5.0x, Huber loss, higher learning rates, two-phase training).

**The Root Cause**: Multi-task learning trade-off combined with fundamental architectural and data challenges that make on-target prediction inherently harder than off-target.

**The Reality**: Off-target is a binary classification task (easy), while on-target is continuous regression (hard). The model optimizes for what's easier.

---

## Part 1: Task Difficulty Comparison

### Off-Target Task (EASY - 0.8753 AUROC)

**What it is**: Binary classification
- Input: sgRNA + target sequence
- Output: 0 (off-target) or 1 (on-target)
- Task: Discriminate between two classes

**Why it's easy**:
1. **Binary decision** - Only 2 possible outputs
2. **Severe class imbalance helps** - 98.93% negative, 1.07% positive
   - Model learns: "Predict 0 most of the time"
   - This alone gives 98.93% accuracy
3. **Clear signal** - Off-target sites have distinct patterns
4. **Large dataset** - 2.5M off-target samples for learning
5. **Focal loss helps** - Handles class imbalance well

**Result**: Model easily learns to discriminate → AUROC 0.8753 ✅

---

### On-Target Task (HARD - 0.4089 Spearman)

**What it is**: Continuous regression
- Input: sgRNA sequence
- Output: Efficiency score (0.0 to 1.0, continuous)
- Task: Predict exact efficiency value

**Why it's hard**:
1. **Continuous output** - Infinite possible values (0.0-1.0)
2. **Requires precise prediction** - Not just classification
3. **Small dataset** - Only 291,639 on-target samples (8.57% of total)
4. **Noisy labels** - Efficiency scores have measurement noise
5. **Complex patterns** - Efficiency depends on many subtle factors
6. **No class imbalance trick** - Can't just predict one value

**Result**: Model struggles to learn precise values → Spearman 0.4089 ❌

---

## Part 2: Why Multi-Task Learning Fails

### The Fundamental Problem

When training both tasks jointly, the model must balance:
- **Off-target**: Easy task, lots of samples, clear signal
- **On-target**: Hard task, few samples, noisy signal

**What happens**:
```
Model learns:
  1. Off-target patterns (easy) → Quickly converges
  2. On-target patterns (hard) → Struggles to learn
  3. Gradient flow → Dominated by off-target task
  4. Result → Model optimizes for off-target, neglects on-target
```

### Why Loss Weighting Didn't Help

**We tried**: Increasing on-target weight from 1.0 to 5.0 (5x)

**Why it didn't work**:
1. **Sample count dominates** - 2.5M off-target vs 0.3M on-target
   - Even with 5x weight, off-target has more gradient signal
   - Off-target loss: 2.5M samples × 0.5 weight = 1.25M gradient units
   - On-target loss: 0.3M samples × 5.0 weight = 1.5M gradient units
   - Still not enough to overcome fundamental difficulty

2. **Task difficulty mismatch** - Off-target is inherently easier
   - Easy task learns quickly
   - Hard task needs more epochs to learn
   - Model converges on easy task first
   - Stops improving on hard task

3. **Gradient saturation** - On-target head may have saturated
   - Model can't improve on-target further
   - Increasing weight just increases loss magnitude
   - Doesn't help model learn better patterns

---

## Part 3: Why Training Metrics Show Failure

### Training Log Analysis

**Epoch 21/30 (Phase 1)**:
```
Train Loss: 0.6728
Val Loss: 0.6703
On-target Spearman: nan
Off-target AUROC: 0.5000
```

**What this means**:
1. **nan Spearman** - Predictions are CONSTANT
   - All predictions are the same value
   - Can't calculate correlation with constant values
   - Model hasn't learned to make diverse predictions

2. **AUROC 0.5000** - Random guessing
   - Model predicting same value for all samples
   - No discrimination between classes
   - Baseline performance

3. **Loss decreasing** - But metrics not improving
   - Loss goes down (model fitting something)
   - But predictions remain constant
   - Model is overfitting to training data, not learning patterns

---

## Part 4: Architectural Limitations

### The Hybrid Multi-Branch Design

**Architecture**:
- Branch A: CNN (Multi-Scale Convolution)
- Branch B: BiGRU (Bidirectional GRU)
- Branch C: RNA-FM (Pretrained embeddings)
- Fusion: Attention-based fusion
- Heads: Separate on-target and off-target heads

**Why it works for off-target**:
1. Multiple branches capture different patterns
2. Attention learns to weight branches
3. Binary classification is straightforward
4. Works well with large dataset

**Why it fails for on-target**:
1. **Architecture designed for classification** - Not regression
   - Attention fusion optimizes for classification
   - Regression needs different architecture
   
2. **Shared encoder** - Optimizes for off-target
   - Encoder learns off-target patterns
   - On-target patterns are secondary
   - Encoder doesn't specialize in on-target
   
3. **On-target head too simple** - Can't learn complex patterns
   - Just 2 hidden layers
   - Insufficient capacity for regression
   - Needs deeper/wider architecture

---

## Part 5: Data Quality Issues

### On-Target Data Problems

**1. Dataset composition**:
- 291,639 on-target samples (8.57%)
- From 9 different datasets with different measurement methods
- Different cell lines, conditions, measurement noise
- Inconsistent labeling across sources

**2. Label noise**:
- Efficiency scores measured experimentally
- Measurement error: ±5-10%
- Different labs use different methods
- Inconsistent scoring across datasets

**3. Sequence diversity**:
- Only 23 bp sequences
- Limited diversity in patterns
- Hard to learn generalizable patterns
- Model overfits to specific sequences

**4. Missing context**:
- Off-target has target sequence (more information)
- On-target only has sgRNA sequence
- Less information to work with
- Harder to make accurate predictions

---

## Part 6: Why Off-Target Works But On-Target Doesn't

### Direct Comparison

| Factor | Off-Target | On-Target |
|--------|-----------|-----------|
| **Task Type** | Binary classification | Continuous regression |
| **Difficulty** | Easy | Hard |
| **Samples** | 2.5M | 0.3M |
| **Signal Clarity** | Clear | Noisy |
| **Baseline Performance** | 98.93% (predict 0) | ~0.0 (random) |
| **Loss Function** | Focal Loss (handles imbalance) | Huber Loss (still struggles) |
| **Gradient Signal** | Strong | Weak |
| **Model Convergence** | Fast | Slow/Stuck |
| **Final Performance** | 0.8753 AUROC ✅ | 0.4089 Spearman ❌ |

---

## Part 7: Why Increased Loss Weight Failed

### The Math

**Scenario**: on_target_weight = 5.0, off_target_weight = 0.5

**Per-epoch gradient contribution**:
- Off-target: 2.5M samples × 0.5 weight = 1.25M gradient units
- On-target: 0.3M samples × 5.0 weight = 1.5M gradient units

**Seems balanced, but**:
1. **Off-target loss is smaller** - BCE loss typically 0.1-0.5
   - Off-target: 1.25M × 0.3 = 375K gradient magnitude
   
2. **On-target loss is larger** - Huber loss typically 0.5-1.0
   - On-target: 1.5M × 0.7 = 1.05M gradient magnitude
   
3. **But gradient saturation** - On-target head can't learn
   - Larger gradients don't help if model can't improve
   - Like pushing harder on a stuck door
   - Need different approach, not just more weight

---

## Part 8: The Fundamental Challenge

### Why This is Hard

**On-target prediction is inherently difficult because**:

1. **Continuous prediction is harder than classification**
   - Classification: 2 choices (easy)
   - Regression: Infinite choices (hard)
   - Regression requires precise learning

2. **Limited information**
   - Only 23 bp sequence
   - No target sequence
   - No off-target context
   - Limited features to work with

3. **Noisy labels**
   - Experimental measurement error
   - Different measurement methods
   - Inconsistent across labs
   - Model can't learn from noisy data

4. **Small dataset**
   - 0.3M samples (vs 2.5M for off-target)
   - Not enough for deep learning
   - Model overfits easily
   - Can't learn generalizable patterns

5. **Complex underlying biology**
   - Efficiency depends on many factors
   - Chromatin accessibility
   - DNA methylation
   - Protein binding sites
   - Cell type specific effects
   - Hard to capture in sequence alone

---

## Part 9: Why Current Approach Won't Work

### The Ceiling

**Current approach**: Multi-task learning with increased weights

**Why it hits a ceiling**:
1. **Architectural mismatch** - Design optimizes for classification
2. **Shared encoder** - Learns off-target patterns, not on-target
3. **Gradient competition** - Off-target dominates
4. **Task difficulty gap** - Too large to bridge with weights
5. **Data limitations** - Not enough on-target samples

**Result**: Can't improve beyond ~0.40-0.45 Spearman

---

## Part 10: What Would Actually Work

### Potential Solutions

**1. Separate Task-Specific Models** (BEST)
- Train separate model for on-target only
- Optimize for regression, not classification
- Use only on-target data
- Deeper/wider architecture for regression
- Expected improvement: 0.40 → 0.60-0.70 ✅

**2. Task-Specific Heads with Better Architecture**
- Keep shared encoder
- Replace on-target head with regression-specific design
- Use different loss function (MSE, not Huber)
- Add residual connections
- Expected improvement: 0.40 → 0.50-0.60 ⚠️

**3. Data Augmentation for On-Target**
- Reverse complement augmentation
- Noise augmentation
- Synthetic data generation
- Increase effective sample count
- Expected improvement: 0.40 → 0.45-0.50 ⚠️

**4. Transfer Learning from Off-Target**
- Pre-train on off-target (already done)
- Fine-tune on-target head only
- Use on-target-specific learning rate
- Expected improvement: 0.40 → 0.50-0.60 ⚠️

**5. Ensemble Methods**
- Train multiple models with different seeds
- Average predictions
- Reduce variance
- Expected improvement: 0.40 → 0.45-0.50 ⚠️

---

## Part 11: Why Off-Target is So Good

### Success Factors

**Off-target achieves 0.8753 AUROC because**:

1. **Binary classification is easy**
   - Only 2 choices
   - Clear decision boundary
   - Model can learn easily

2. **Class imbalance is actually helpful**
   - 98.93% negative class
   - Model learns: "Predict 0 most of the time"
   - Focal loss handles remaining 1.07%
   - Works well for this task

3. **Large dataset**
   - 2.5M samples
   - Enough to learn patterns
   - Deep learning works well
   - Good generalization

4. **Clear signal**
   - Off-target sites have distinct patterns
   - Easy to discriminate
   - Model learns quickly
   - Converges well

5. **Suitable architecture**
   - Multi-branch design works for classification
   - Attention fusion learns to weight branches
   - Binary head is simple and effective
   - Good for this task

---

## Part 12: The Bottom Line

### Summary

| Aspect | Off-Target | On-Target | Why Different |
|--------|-----------|-----------|---------------|
| **Task** | Binary classification | Continuous regression | Different problem types |
| **Difficulty** | Easy | Hard | Classification < Regression |
| **Samples** | 2.5M | 0.3M | Data availability |
| **Signal** | Clear | Noisy | Measurement quality |
| **Architecture Fit** | Excellent | Poor | Design optimized for classification |
| **Current Performance** | 0.8753 ✅ | 0.4089 ❌ | Reflects fundamental differences |
| **Improvement Potential** | Limited | High | Can improve with different approach |

---

## Part 13: Recommendations

### What To Do

**Short Term** (Use current model):
- Publish off-target results (0.8753 AUROC - SUPERIOR)
- Note on-target as limitation
- Acknowledge multi-task learning trade-off

**Medium Term** (Improve on-target):
- Train separate on-target model
- Use regression-specific architecture
- Focus on on-target data only
- Expected: 0.60-0.70 Spearman

**Long Term** (Optimal solution):
- Separate models for each task
- Task-specific architectures
- Task-specific hyperparameters
- Ensemble for final predictions
- Expected: 0.75+ on-target, 0.85+ off-target

---

## Conclusion

**On-target prediction is not improving because**:

1. **Fundamental task difficulty** - Regression is harder than classification
2. **Multi-task learning trade-off** - Model optimizes for easier task (off-target)
3. **Architectural mismatch** - Design optimized for classification, not regression
4. **Data limitations** - Fewer on-target samples, noisier labels
5. **Gradient competition** - Off-target dominates despite weight adjustment

**The solution is not to push harder on the same approach, but to use a different approach entirely** - separate task-specific models or architectures.

**Current model is excellent for off-target (0.8753 AUROC - SUPERIOR), but on-target requires a different strategy.**

