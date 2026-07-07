# CRISPR-UniPredict Execution Checklist

Complete checklist for setting up, executing, and validating the CRISPR-UniPredict pipeline.

---

## Pre-Execution Checklist

### Environment Setup

- [ ] Python 3.9+ installed
- [ ] Conda/venv environment created
- [ ] Environment activated
- [ ] `pip install -r requirements.txt` completed successfully
- [ ] `pip install -r web_app/requirements.txt` completed successfully
- [ ] No dependency conflicts reported
- [ ] GPU detected (if available)
  - [ ] CUDA 11.8+ installed
  - [ ] cuDNN installed
  - [ ] `nvidia-smi` shows GPU
  - [ ] PyTorch can access GPU: `python -c "import torch; print(torch.cuda.is_available())"`

**Verification Command:**
```bash
python scripts/validate_system.py
```

Expected output: ✓ All checks passed

---

### Data Preparation

#### CRISPR_HNN Datasets (9 datasets)

- [ ] WT (Wild-Type) dataset downloaded
- [ ] ESP (Enhanced Specificity) dataset downloaded
- [ ] HF (High-Fidelity) dataset downloaded
- [ ] xCas9 dataset downloaded
- [ ] SpCas9 dataset downloaded
- [ ] Sniper dataset downloaded
- [ ] HCT116 dataset downloaded
- [ ] HELA dataset downloaded
- [ ] HL60 dataset downloaded

#### CCLMoff Datasets (13 methods)

- [ ] CCLMoff dataset downloaded
- [ ] All 13 prediction methods included
- [ ] Dataset size verified (>700MB)

#### Data Processing

- [ ] Data inspection completed
  ```bash
  python -c "import pandas as pd; df = pd.read_csv('data/raw/crispr_hnn/WT.csv'); print(df.head())"
  ```

- [ ] All datasets converted to unified format
  ```bash
  python scripts/preprocess_data.py --input data/raw --output data/processed
  ```

- [ ] Train/val/test splits created
  - [ ] Train set: 70% of data
  - [ ] Validation set: 15% of data
  - [ ] Test set: 15% of data

- [ ] Data validation passed
  ```bash
  python scripts/validate_data.py --data_dir data/processed
  ```

- [ ] No missing values
- [ ] No duplicate sequences
- [ ] All sequences valid (ACGT only)
- [ ] Label ranges correct (0-1)

**Verification:**
```bash
# Check dataset sizes
ls -lh data/processed/*/
# Should show train.csv, val.csv, test.csv for each dataset
```

---

### Model Setup

#### Pretrained Models

- [ ] RNA-FM model downloaded
  ```bash
  # Download from: https://github.com/ml4bio/RNA-FM
  # Place at: models/pretrained/rna_fm_t12.pt
  ```

- [ ] Model checkpoint verified
  ```bash
  python -c "import torch; ckpt = torch.load('models/pretrained/rna_fm_t12.pt'); print(ckpt.keys())"
  ```

#### Model Components

- [ ] MultiScaleConvolution (MSC) implemented
- [ ] BiGRU module implemented
- [ ] MultiHeadSelfAttention (MHSA) implemented
- [ ] RNA-FM encoder implemented
- [ ] AttentionFusion module implemented
- [ ] CRISPRUniPredict main model implemented

#### Configuration

- [ ] `configs/model_config.yaml` created
  ```bash
  python -c "from configs.config_loader import ConfigLoader; cfg = ConfigLoader.load('configs/model_config.yaml'); print(cfg)"
  ```

- [ ] All hyperparameters set correctly
  - [ ] Sequence length: 23 bp
  - [ ] MSC channels: 64
  - [ ] MHSA embedding: 256
  - [ ] BiGRU hidden: 128
  - [ ] Dropout: 0.35

#### Testing

- [ ] Unit tests passed
  ```bash
  pytest tests/test_models.py -v
  ```
  Expected: All tests pass

- [ ] Integration tests passed
  ```bash
  pytest tests/test_integration.py -v
  ```
  Expected: All tests pass

- [ ] Code coverage >80%
  ```bash
  pytest tests/ --cov=models --cov=utils --cov-report=term-missing
  ```

---

### Training Preparation

#### Data Loading

- [ ] DataLoader factory implemented
  ```bash
  python -c "from utils.preprocessing.dataloader_factory import DataLoaderFactory; print('✓ DataLoader factory OK')"
  ```

- [ ] Batch collation tested
  ```bash
  pytest tests/test_models.py::TestDataset -v
  ```

- [ ] Samplers implemented (Bootstrap, Weighted, Stratified)
  ```bash
  pytest tests/test_models.py -k "sampler" -v
  ```

#### Loss Functions

- [ ] MultiTaskLoss implemented
- [ ] FocalLoss implemented
- [ ] WeightedMultiTaskLoss implemented
- [ ] Loss computation tested
  ```bash
  pytest tests/test_models.py::TestMetricsCalculator -v
  ```

#### Metrics

- [ ] MetricsCalculator implemented
- [ ] On-target metrics: Spearman, Pearson, MAE, RMSE
- [ ] Off-target metrics: AUROC, F1, Balanced Accuracy
- [ ] Metrics tested
  ```bash
  pytest tests/test_models.py::TestMetricsCalculator -v
  ```

#### Trainer

- [ ] Trainer class implemented
- [ ] Mixed precision training (AMP) configured
- [ ] Gradient clipping configured
- [ ] Early stopping configured
- [ ] Checkpoint management configured
- [ ] Logging configured (TensorBoard/WandB)

#### Optimization

- [ ] Optimizer setup implemented
  - [ ] Differential learning rates configured
  - [ ] RNA-FM: 5e-4
  - [ ] Feature extraction: 1e-3
  - [ ] Heads: 1e-3

- [ ] Scheduler setup implemented
  - [ ] Linear warmup: 5 epochs
  - [ ] ReduceLROnPlateau configured

---

## Execution Phases

### Phase 1: Small-Scale Test (Debug Mode)

**Objective:** Verify training pipeline works without errors

**Duration:** ~30 minutes

**Steps:**

1. [ ] Create small test dataset (1000 samples)
   ```bash
   python scripts/create_test_dataset.py --size 1000 --output data/test_small
   ```

2. [ ] Start training
   ```bash
   python scripts/train.py \
     --config configs/model_config.yaml \
     --train_data data/test_small/train.csv \
     --val_data data/test_small/val.csv \
     --epochs 2 \
     --batch_size 32 \
     --experiment_name debug_test
   ```

3. [ ] Monitor training
   - [ ] No CUDA out of memory errors
   - [ ] No NaN losses
   - [ ] Loss decreasing
   - [ ] Training speed: ~100 samples/sec

4. [ ] Verify checkpoints
   ```bash
   ls -lh models/checkpoints/
   # Should see: latest.pt, best.pt
   ```

5. [ ] Check logs
   ```bash
   tail -f logs/debug_test_*/training.log
   ```

6. [ ] Verify memory usage
   - [ ] GPU memory: <8GB
   - [ ] RAM: <16GB

**Success Criteria:**
- ✓ Training completes without errors
- ✓ Loss decreases over epochs
- ✓ Checkpoints saved
- ✓ Memory usage reasonable

---

### Phase 2: Medium-Scale Validation

**Objective:** Validate training on 10% of data

**Duration:** ~3-4 hours

**Steps:**

1. [ ] Create 10% dataset
   ```bash
   python scripts/create_subset_dataset.py --fraction 0.1 --output data/subset_10pct
   ```

2. [ ] Start training
   ```bash
   python scripts/train.py \
     --config configs/model_config.yaml \
     --train_data data/subset_10pct/train.csv \
     --val_data data/subset_10pct/val.csv \
     --epochs 5 \
     --batch_size 32 \
     --experiment_name validation_10pct
   ```

3. [ ] Monitor metrics
   - [ ] Training loss decreasing
   - [ ] Validation loss decreasing
   - [ ] Spearman correlation improving
   - [ ] AUROC improving

4. [ ] Check for overfitting
   - [ ] Training loss < Validation loss (acceptable)
   - [ ] Gap not too large (< 0.1)

5. [ ] Evaluate on validation set
   ```bash
   python scripts/evaluate.py \
     --checkpoint models/checkpoints/best.pt \
     --test_data data/subset_10pct/val.csv \
     --output_dir results/validation_10pct
   ```

6. [ ] Review results
   ```bash
   cat results/validation_10pct/metrics.json
   ```

**Success Criteria:**
- ✓ Training completes successfully
- ✓ Metrics improving over epochs
- ✓ No overfitting
- ✓ Validation metrics reasonable

---

### Phase 3: Full Training

**Objective:** Train on complete dataset

**Duration:** ~24-48 hours

**Steps:**

1. [ ] Prepare full dataset
   ```bash
   python scripts/prepare_full_dataset.py --output data/processed/combined
   ```

2. [ ] Verify dataset sizes
   ```bash
   wc -l data/processed/combined/*.csv
   # Expected: ~100k samples total
   ```

3. [ ] Start training
   ```bash
   python scripts/train.py \
     --config configs/model_config.yaml \
     --train_data data/processed/combined/train.csv \
     --val_data data/processed/combined/val.csv \
     --test_data data/processed/combined/test.csv \
     --epochs 50 \
     --batch_size 64 \
     --experiment_name full_training \
     --use_wandb
   ```

4. [ ] Monitor training (in separate terminal)
   ```bash
   # Option 1: TensorBoard
   tensorboard --logdir logs/
   
   # Option 2: WandB
   # Check at: https://wandb.ai/yourname/crispr-unipredict
   ```

5. [ ] Check training progress
   - [ ] Loss decreasing smoothly
   - [ ] No NaN/Inf values
   - [ ] GPU memory stable
   - [ ] Training speed: 50-100 samples/sec

6. [ ] Monitor checkpoints
   ```bash
   watch -n 60 'ls -lh models/checkpoints/'
   ```

7. [ ] Early stopping
   - [ ] Monitor validation loss
   - [ ] Stop if no improvement for 10 epochs

**Success Criteria:**
- ✓ Training completes without errors
- ✓ Best model saved
- ✓ Training curves smooth
- ✓ Final metrics reasonable

---

### Phase 4: Evaluation

**Objective:** Comprehensive evaluation on test set

**Duration:** ~2-3 hours

**Steps:**

1. [ ] Evaluate on test set
   ```bash
   python scripts/evaluate.py \
     --checkpoint models/checkpoints/best.pt \
     --test_data data/processed/combined/test.csv \
     --output_dir results/evaluation
   ```

2. [ ] Compute all metrics
   - [ ] On-target: Spearman, Pearson, MAE, RMSE
   - [ ] Off-target: AUROC, AUPRC, F1, Balanced Accuracy
   - [ ] Comprehensive score

3. [ ] Generate visualizations
   - [ ] ROC curve
   - [ ] PR curve
   - [ ] Scatter plots (predicted vs actual)
   - [ ] Correlation heatmaps

4. [ ] Review results
   ```bash
   cat results/evaluation/metrics.json
   ```

5. [ ] Compare with baselines
   ```bash
   python scripts/compare_baselines.py \
     --model_checkpoint models/checkpoints/best.pt \
     --test_data data/processed/combined/test.csv \
     --output_dir results/baseline_comparison
   ```

**Success Criteria:**
- ✓ All metrics computed
- ✓ Visualizations generated
- ✓ Better than baselines
- ✓ Results reasonable

---

### Phase 5: Analysis

**Objective:** Detailed analysis and interpretation

**Duration:** ~5-6 hours

**Steps:**

1. [ ] Ablation study
   ```bash
   python scripts/ablation_study.py \
     --config configs/model_config.yaml \
     --train_data data/processed/combined/train.csv \
     --val_data data/processed/combined/val.csv \
     --test_data data/processed/combined/test.csv \
     --output_dir results/ablation
   ```

2. [ ] Cross-dataset validation
   ```bash
   python scripts/cross_dataset_validation.py \
     --config configs/model_config.yaml \
     --output_dir results/cross_validation
   ```

3. [ ] Statistical analysis
   ```bash
   python scripts/statistical_analysis.py \
     --predictions results/evaluation/predictions.csv \
     --output_dir results/statistical_analysis
   ```

4. [ ] Attention visualization
   ```bash
   python scripts/visualize_attention.py \
     --checkpoint models/checkpoints/best.pt \
     --sequences data/examples.csv \
     --output_dir results/attention_viz
   ```

5. [ ] Feature importance
   ```bash
   python scripts/analyze_features.py \
     --checkpoint models/checkpoints/best.pt \
     --test_data data/processed/combined/test.csv \
     --output_dir results/feature_importance
   ```

**Success Criteria:**
- ✓ All analyses completed
- ✓ Results consistent
- ✓ Insights documented

---

### Phase 6: Documentation

**Objective:** Generate paper materials and documentation

**Duration:** ~8-10 hours

**Steps:**

1. [ ] Generate results summary
   ```bash
   python scripts/generate_results_summary.py \
     --results_dir results/evaluation \
     --output_dir results/paper_materials
   ```

2. [ ] Generate all tables
   - [ ] Dataset statistics
   - [ ] Model architecture
   - [ ] Main results
   - [ ] Cross-dataset generalization
   - [ ] Ablation study

3. [ ] Generate all figures
   - [ ] Architecture diagram
   - [ ] Performance comparison
   - [ ] ROC and PR curves
   - [ ] Correlation plots
   - [ ] Attention visualization
   - [ ] Cross-dataset heatmap
   - [ ] Score distribution

4. [ ] Generate paper draft
   ```bash
   python scripts/generate_paper_draft.py \
     --results_dir results/evaluation \
     --output_dir paper/
   ```

5. [ ] Create tutorials
   - [ ] Review all Jupyter notebooks
   - [ ] Verify all examples work
   - [ ] Test on clean environment

6. [ ] Create API documentation
   - [ ] Review API reference
   - [ ] Test all endpoints
   - [ ] Verify examples work

**Success Criteria:**
- ✓ All tables generated
- ✓ All figures generated
- ✓ Paper draft created
- ✓ Tutorials working
- ✓ Documentation complete

---

## Post-Execution

### Results Validation

- [ ] All metrics computed correctly
- [ ] Results better than baselines
- [ ] Cross-dataset validation passed
- [ ] Statistical tests significant
- [ ] Ablation study shows component importance

### Code Quality

- [ ] All tests passing
  ```bash
  pytest tests/ -v
  ```

- [ ] Code coverage >80%
  ```bash
  pytest tests/ --cov --cov-report=term-missing
  ```

- [ ] No linting errors
  ```bash
  flake8 models/ utils/ scripts/ --max-line-length=100
  ```

- [ ] Code formatted
  ```bash
  black models/ utils/ scripts/
  ```

### Repository Cleanup

- [ ] Remove temporary files
  ```bash
  rm -rf __pycache__ .pytest_cache .coverage
  ```

- [ ] Remove large intermediate files
  ```bash
  rm -rf data/processed/intermediate/
  ```

- [ ] Verify important files present
  - [ ] models/checkpoints/best.pt
  - [ ] results/evaluation/metrics.json
  - [ ] paper/paper_draft.tex
  - [ ] notebooks/*.ipynb

### Deployment

- [ ] Docker image built
  ```bash
  docker build -t crispr-unipredict:latest .
  ```

- [ ] Docker image tested
  ```bash
  docker run -p 8000:8000 -p 8501:8501 crispr-unipredict:latest
  ```

- [ ] Web app deployed
  ```bash
  streamlit run web_app/app.py
  ```

- [ ] API deployed
  ```bash
  uvicorn api.inference_api:app --host 0.0.0.0 --port 8000
  ```

### GitHub Publication

- [ ] Repository initialized
  ```bash
  git init
  git add .
  git commit -m "Initial commit: CRISPR-UniPredict"
  ```

- [ ] .gitignore configured
- [ ] README.md reviewed
- [ ] LICENSE added
- [ ] Repository pushed to GitHub
- [ ] GitHub Pages configured (optional)

### Final Verification

- [ ] System validation passed
  ```bash
  python scripts/validate_system.py
  ```

- [ ] All documentation complete
- [ ] All examples working
- [ ] All tests passing
- [ ] Ready for publication

---

## Estimated Timeline

### Setup and Testing
- Environment setup: 1-2 hours
- Data preparation: 1-2 hours
- Model setup: 1-2 hours
- **Subtotal: 4-6 hours**

### Training and Evaluation
- Phase 1 (Debug): 0.5 hours
- Phase 2 (Validation): 3-4 hours
- Phase 3 (Full training): 24-48 hours
- Phase 4 (Evaluation): 2-3 hours
- **Subtotal: 30-56 hours**

### Analysis and Documentation
- Phase 5 (Analysis): 5-6 hours
- Phase 6 (Documentation): 8-10 hours
- **Subtotal: 13-16 hours**

### Post-Execution
- Code quality: 1-2 hours
- Deployment: 1-2 hours
- GitHub publication: 1 hour
- **Subtotal: 3-5 hours**

### **TOTAL ESTIMATED TIME: 50-83 hours**

---

## Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
- Solution: Reduce batch size in config
- Or: Use CPU (slower but works)

**Issue: Training too slow**
- Solution: Check GPU utilization with `nvidia-smi`
- Or: Increase batch size (if memory allows)

**Issue: NaN loss**
- Solution: Reduce learning rate
- Or: Check data for invalid values

**Issue: Validation loss not improving**
- Solution: Train longer (increase epochs)
- Or: Adjust learning rate schedule

**Issue: Tests failing**
- Solution: Run `pytest tests/ -v` to see details
- Or: Check environment setup

---

## Success Indicators

✓ **Phase 1:** Training completes without errors
✓ **Phase 2:** Metrics improving on validation set
✓ **Phase 3:** Best model saved after full training
✓ **Phase 4:** Test metrics better than baselines
✓ **Phase 5:** Ablation study shows component importance
✓ **Phase 6:** All paper materials generated
✓ **Final:** System validation passed, ready for publication

---

## Next Steps After Completion

1. Submit paper to journal/conference
2. Create GitHub releases
3. Publish on PyPI (optional)
4. Deploy web service
5. Gather community feedback
6. Plan future improvements

---

**Good luck with your CRISPR-UniPredict execution! 🚀**
