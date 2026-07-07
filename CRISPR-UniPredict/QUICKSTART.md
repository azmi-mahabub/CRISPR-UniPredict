# CRISPR-UniPredict Quick Start Guide

## Installation

1. Ensure all dependencies are installed:
```bash
pip install torch numpy pandas scikit-learn
```

2. (Optional) Install RNA-FM for enhanced embeddings:
```bash
# Add RNA-FM to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/RNA-FM-main
```

---

## Basic Usage

### 1. Initialize the Model

```python
import torch
from models.crispr_unipredict import CRISPRUniPredict

# Create model
model = CRISPRUniPredict(
    seq_len=23,
    hidden_dim=256,
    dropout=0.35,
    device='cuda'  # or 'cpu'
)

print(model.get_model_info())
```

### 2. Prepare Input Data

```python
from models.encoding import SequenceEncoder

# Initialize encoder
encoder = SequenceEncoder(device='cuda')

# Example sequences
sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"  # 23bp sgRNA
target = "GCTAGCTAGCTAGCTAGCTAGCT"  # 23bp target

# Encode sequences
sgrna_onehot = encoder.encode_one_hot(sgrna)  # (1, 23, 4)
sgrna_label = encoder.encode_label(sgrna)     # (1, 23)

print(f"One-hot shape: {sgrna_onehot.shape}")
print(f"Label shape: {sgrna_label.shape}")
```

### 3. Make Predictions

```python
# Set model to evaluation mode
model.eval()

with torch.no_grad():
    # On-target prediction
    on_target_score = model.predict_on_target(sgrna_onehot, sgrna_label)
    print(f"On-target score: {on_target_score.item():.4f}")
    
    # Off-target prediction
    off_target_prob = model.predict_off_target(sgrna_onehot, sgrna_label)
    print(f"Off-target probability: {off_target_prob.item():.4f}")
    
    # Both predictions
    on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
    print(f"On-target: {on_target.item():.4f}, Off-target: {off_target.item():.4f}")
```

---

## Batch Processing

```python
import torch
from models.encoding import SequenceEncoder

# Prepare batch of sequences
sequences = [
    "GCTAGCTAGCTAGCTAGCTAGCT",
    "ATGCATGCATGCATGCATGCATG",
    "CCGGCCGGCCGGCCGGCCGGCCG",
]

encoder = SequenceEncoder(device='cuda')

# Encode batch
batch_onehot = []
batch_label = []

for seq in sequences:
    onehot = encoder.encode_one_hot(seq)
    label = encoder.encode_label(seq)
    batch_onehot.append(onehot)
    batch_label.append(label)

# Stack into batch
batch_onehot = torch.cat(batch_onehot, dim=0)  # (3, 23, 4)
batch_label = torch.cat(batch_label, dim=0)    # (3, 23)

# Predict
model.eval()
with torch.no_grad():
    on_target, off_target = model(batch_onehot, batch_label, task_type='both')

print(f"Predictions shape: {on_target.shape}")
print(f"On-target scores: {on_target.squeeze().tolist()}")
print(f"Off-target probs: {off_target.squeeze().tolist()}")
```

---

## Training

### Single-Task Training (On-Target Only)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.crispr_unipredict import CRISPRUniPredict

# Initialize model and optimizer
model = CRISPRUniPredict(device='cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (sgrna_onehot, sgrna_label, on_target_label) in enumerate(train_loader):
        # Move to device
        sgrna_onehot = sgrna_onehot.to('cuda')
        sgrna_label = sgrna_label.to('cuda')
        on_target_label = on_target_label.to('cuda')
        
        # Forward pass
        on_target_pred = model.predict_on_target(sgrna_onehot, sgrna_label)
        
        # Compute loss
        loss = criterion(on_target_pred, on_target_label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

### Multi-Task Training (On-Target + Off-Target)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.crispr_unipredict import CRISPRUniPredict

# Initialize model and optimizer
model = CRISPRUniPredict(device='cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loss functions
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# Training loop
num_epochs = 50
on_target_weight = 1.0
off_target_weight = 0.5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (sgrna_onehot, sgrna_label, on_target_label, off_target_label) in enumerate(train_loader):
        # Move to device
        sgrna_onehot = sgrna_onehot.to('cuda')
        sgrna_label = sgrna_label.to('cuda')
        on_target_label = on_target_label.to('cuda')
        off_target_label = off_target_label.to('cuda')
        
        # Forward pass
        on_target_pred, off_target_pred = model(sgrna_onehot, sgrna_label, task_type='both')
        
        # Compute losses
        loss_on_target = mse_loss(on_target_pred, on_target_label)
        loss_off_target = bce_loss(off_target_pred, off_target_label)
        
        # Combined loss
        loss = on_target_weight * loss_on_target + off_target_weight * loss_off_target
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

---

## Selective Branch Training

### Freeze Pretrained Components

```python
# Freeze RNA-FM (Branch C) - keep it as feature extractor
model.freeze_branch_c()

# Train only Branches A and B
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3
)

# Check trainable parameters
info = model.get_model_info()
print(f"Trainable parameters: {info['trainable_parameters']:,}")
```

### Progressive Training

```python
# Phase 1: Train only Branch A (CNN)
model.freeze_branch_b()
model.freeze_branch_c()
optimizer_a = optim.Adam(model.msc.parameters(), lr=1e-3)

# ... train for N epochs ...

# Phase 2: Unfreeze and train all branches
model.unfreeze_branch_b()
model.unfreeze_branch_c()
optimizer_all = optim.Adam(model.parameters(), lr=1e-4)

# ... continue training ...
```

---

## Evaluation

### Compute Metrics

```python
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score

model.eval()

all_on_target_pred = []
all_on_target_true = []
all_off_target_pred = []
all_off_target_true = []

with torch.no_grad():
    for sgrna_onehot, sgrna_label, on_target_label, off_target_label in test_loader:
        sgrna_onehot = sgrna_onehot.to('cuda')
        sgrna_label = sgrna_label.to('cuda')
        
        on_target_pred, off_target_pred = model(sgrna_onehot, sgrna_label, task_type='both')
        
        all_on_target_pred.extend(on_target_pred.cpu().numpy())
        all_on_target_true.extend(on_target_label.numpy())
        all_off_target_pred.extend(off_target_pred.cpu().numpy())
        all_off_target_true.extend(off_target_label.numpy())

# Convert to numpy
on_target_pred = np.array(all_on_target_pred).flatten()
on_target_true = np.array(all_on_target_true).flatten()
off_target_pred = np.array(all_off_target_pred).flatten()
off_target_true = np.array(all_off_target_true).flatten()

# Compute metrics
on_target_mse = mean_squared_error(on_target_true, on_target_pred)
on_target_rmse = np.sqrt(on_target_mse)

off_target_auc = roc_auc_score(off_target_true, off_target_pred)
off_target_acc = accuracy_score(off_target_true, (off_target_pred > 0.5).astype(int))

print(f"On-target RMSE: {on_target_rmse:.4f}")
print(f"Off-target AUC: {off_target_auc:.4f}")
print(f"Off-target Accuracy: {off_target_acc:.4f}")
```

---

## Saving and Loading

### Save Model

```python
import torch

# Save model weights
torch.save(model.state_dict(), 'crispr_unipredict_weights.pth')

# Save entire model
torch.save(model, 'crispr_unipredict_model.pth')
```

### Load Model

```python
import torch
from models.crispr_unipredict import CRISPRUniPredict

# Load weights
model = CRISPRUniPredict(device='cuda')
model.load_state_dict(torch.load('crispr_unipredict_weights.pth'))

# Or load entire model
model = torch.load('crispr_unipredict_model.pth')
```

---

## Common Issues and Solutions

### Issue: RNA-FM not available
**Solution**: Model gracefully falls back to embedding + projection. No action needed.

### Issue: Out of memory
**Solution**: Reduce batch size or use gradient accumulation
```python
# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    # ... forward pass ...
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: Poor convergence
**Solution**: Try different learning rates or adjust loss weights
```python
# Try lower learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Adjust task weights
loss = 1.0 * loss_on_target + 0.3 * loss_off_target
```

### Issue: Model overfitting
**Solution**: Increase dropout or use early stopping
```python
# Increase dropout
model = CRISPRUniPredict(dropout=0.5, device='cuda')

# Early stopping
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

---

## Performance Tips

1. **Use GPU**: 10-20x faster than CPU
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = CRISPRUniPredict(device=device)
   ```

2. **Batch Processing**: Process multiple sequences at once
   - Batch size 32: ~50-100ms (CPU), ~10-20ms (GPU)

3. **Mixed Precision**: Use automatic mixed precision for faster training
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       loss = criterion(output, target)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   ```

4. **Gradient Checkpointing**: Reduce memory usage
   ```python
   # Checkpoint specific layers
   from torch.utils.checkpoint import checkpoint
   output = checkpoint(model.mhsa, branch_a)
   ```

---

## Next Steps

1. **Prepare your data**: Format sequences and labels
2. **Train the model**: Use the training examples above
3. **Evaluate**: Compute metrics on test set
4. **Fine-tune**: Adjust hyperparameters for your specific task
5. **Deploy**: Save and load model for inference

For more details, see `CRISPR_UNIPREDICT_ARCHITECTURE.md`
