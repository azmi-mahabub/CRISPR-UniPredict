# Edits Made During 2026-04-19 Training Session

**Session Date:** April 19, 2026  
**Purpose:** Fixed dataloader validation performance and configured RNA-FM parameter handling  
**Status:** All edits have been REVERTED to original working state

---

## Summary of Changes

This document tracks all file edits made during this session for easy tracking and potential undoing.

### Timeline:
1. ✅ **Applied:** Optimized dataloader validation (vectorized)
2. ✅ **Applied:** Added `use_rna_fm` parameter to model
3. ✅ **Applied:** Updated train.py to read and pass `use_rna_fm` from config
4. ✅ **REVERTED:** All changes reverted back to original working state

---

## Detailed Edit Log

### Edit #1: Dataloader Validation Optimization (REVERTED)

**File:** `utils/preprocessing/dataloader_fast.py`  
**Lines:** 47-54  
**Type:** Performance optimization  
**Reason:** Original function-based validation was slow on large datasets, causing timeouts during dataloader initialization  
**Status:** ✅ REVERTED

**Original Code (CURRENT - after revert):**
```python
        # Filter valid sequences - must be 22-24 bp with valid nucleotides
        valid_nucleotides = set('ACGTU')
        def is_valid(seq):
            if not isinstance(seq, str):
                return False
            # Must be 22-24 bp and contain only valid nucleotides
            return 22 <= len(seq) <= 24 and all(c.upper() in valid_nucleotides for c in seq)
        
        mask = self.df['sgrna'].apply(is_valid) & self.df['target'].apply(is_valid)
```

**Modified Code (REMOVED):**
```python
        # Filter valid sequences - must be 22-24 bp with valid nucleotides (vectorized for speed)
        import re
        # Vectorized validation: check length first (fast), then regex for valid nucleotides
        sgrna_valid = (self.df['sgrna'].astype(str).str.len() >= 22) & \
                      (self.df['sgrna'].astype(str).str.len() <= 24) & \
                      (self.df['sgrna'].astype(str).str.match(r'^[ACGTUacgtu]+$'))
        
        target_valid = (self.df['target'].astype(str).str.len() >= 22) & \
                       (self.df['target'].astype(str).str.len() <= 24) & \
                       (self.df['target'].astype(str).str.match(r'^[ACGTUacgtu]+$'))
        
        mask = sgrna_valid & target_valid
```

---

### Edit #2: Model Constructor - Add `use_rna_fm` Parameter (REVERTED)

**File:** `models/crispr_unipredict.py`  
**Lines:** 110-122  
**Type:** Parameter addition to constructor  
**Reason:** Attempted to allow disabling RNA-FM via config for faster training  
**Status:** ✅ REVERTED

**Original Code (CURRENT - after revert):**
```python
    def __init__(self,
                 seq_len: int = 23,
                 msc_out_channels: int = 64,
                 mhsa_embed_dim: int = 256,
                 bigru_hidden_dim: int = 128,
                 embedding_dim: int = 128,
                 vocab_size: int = 6,
                 rna_fm_path: str = 'models/pretrained/rna_fm_t12.pt',
                 hidden_dim: int = 256,
                 dropout: float = 0.35,
                 device: str = 'cpu',
                 freeze_rna_fm: bool = True,
                 num_unfreeze_rna_fm: int = 2):
```

**Modified Code (REMOVED):**
```python
    def __init__(self,
                 seq_len: int = 23,
                 msc_out_channels: int = 64,
                 mhsa_embed_dim: int = 256,
                 bigru_hidden_dim: int = 128,
                 embedding_dim: int = 128,
                 vocab_size: int = 6,
                 rna_fm_path: str = 'models/pretrained/rna_fm_t12.pt',
                 hidden_dim: int = 256,
                 dropout: float = 0.35,
                 device: str = 'cpu',
                 use_rna_fm: bool = True,  # <-- ADDED
                 freeze_rna_fm: bool = True,
                 num_unfreeze_rna_fm: int = 2):
```

---

### Edit #3: Model Docstring Update (REVERTED)

**File:** `models/crispr_unipredict.py`  
**Lines:** 124-138  
**Type:** Documentation update  
**Reason:** Added documentation for new `use_rna_fm` parameter  
**Status:** ✅ REVERTED

**Original Code (CURRENT - after revert):**
```python
        """
        Initialize CRISPR-UniPredict model
        
        Args:
            seq_len: Sequence length (default: 23 for sgRNA)
            msc_out_channels: Output channels per MSC branch (default: 64)
            mhsa_embed_dim: MHSA embedding dimension (default: 256)
            bigru_hidden_dim: BiGRU hidden dimension (default: 128)
            embedding_dim: Label embedding dimension (default: 128)
            vocab_size: Vocabulary size for label encoding (default: 6)
            rna_fm_path: Path to pretrained RNA-FM model
            hidden_dim: Hidden dimension for fusion and task heads (default: 256)
            dropout: Dropout rate (default: 0.35)
            device: Device to use ('cpu' or 'cuda')
            freeze_rna_fm: Whether to freeze RNA-FM layers (default: True)
            num_unfreeze_rna_fm: Number of RNA-FM layers to unfreeze (default: 2)
        """
```

**Modified Code (REMOVED):**
```python
        """
        Initialize CRISPR-UniPredict model
        
        Args:
            seq_len: Sequence length (default: 23 for sgRNA)
            msc_out_channels: Output channels per MSC branch (default: 64)
            mhsa_embed_dim: MHSA embedding dimension (default: 256)
            bigru_hidden_dim: BiGRU hidden dimension (default: 128)
            embedding_dim: Label embedding dimension (default: 128)
            vocab_size: Vocabulary size for label encoding (default: 6)
            rna_fm_path: Path to pretrained RNA-FM model
            hidden_dim: Hidden dimension for fusion and task heads (default: 256)
            dropout: Dropout rate (default: 0.35)
            device: Device to use ('cpu' or 'cuda')
            use_rna_fm: Whether to use RNA-FM (default: True). Set to False for faster training  # <-- ADDED
            freeze_rna_fm: Whether to freeze RNA-FM layers (default: True)
            num_unfreeze_rna_fm: Number of RNA-FM layers to unfreeze (default: 2)
        """
```

---

### Edit #4: Model RNA-FM Initialization Logic (REVERTED)

**File:** `models/crispr_unipredict.py`  
**Lines:** 186-205  
**Type:** Conditional initialization  
**Reason:** Made RNA-FM initialization conditional based on `use_rna_fm` parameter  
**Status:** ✅ REVERTED

**Original Code (CURRENT - after revert):**
```python
        # ============ BRANCH C: Label → RNA-FM Encoder ============
        try:
            self.rna_fm = RNAFMEncoder(
                model_path=rna_fm_path,
                freeze_layers=freeze_rna_fm,
                num_unfreeze_layers=num_unfreeze_rna_fm,
                device=device
            )
            self.rna_fm_available = True
            rna_fm_out_dim = 640  # RNA-FM embedding dimension
            # If forward() is called without raw sequences, approximate branch C from embeddings
            self.branch_c_embed_proj = nn.Linear(embedding_dim, rna_fm_out_dim)
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            warnings.warn(f"RNA-FM not available: {str(e)}. Using placeholder.")
            self.rna_fm_available = False
            rna_fm_out_dim = 256  # Fallback dimension
            self.rna_fm_fallback = nn.Linear(embedding_dim, rna_fm_out_dim)
```

**Modified Code (REMOVED):**
```python
        # ============ BRANCH C: Label → RNA-FM Encoder ============
        if use_rna_fm:  # <-- ADDED conditional
            try:
                self.rna_fm = RNAFMEncoder(
                    model_path=rna_fm_path,
                    freeze_layers=freeze_rna_fm,
                    num_unfreeze_layers=num_unfreeze_rna_fm,
                    device=device
                )
                self.rna_fm_available = True
                rna_fm_out_dim = 640  # RNA-FM embedding dimension
                # If forward() is called without raw sequences, approximate branch C from embeddings
                self.branch_c_embed_proj = nn.Linear(embedding_dim, rna_fm_out_dim)
            except (ImportError, FileNotFoundError, RuntimeError) as e:
                warnings.warn(f"RNA-FM not available: {str(e)}. Using placeholder.")
                self.rna_fm_available = False
                rna_fm_out_dim = 256  # Fallback dimension
                self.rna_fm_fallback = nn.Linear(embedding_dim, rna_fm_out_dim)
        else:  # <-- ADDED else block
            # RNA-FM disabled via config - use fallback embeddings
            self.rna_fm_available = False
            rna_fm_out_dim = 256  # Fallback dimension
            # Create a simple projection layer as fallback
            self.rna_fm_fallback = nn.Linear(embedding_dim, rna_fm_out_dim)
```

---

### Edit #5: Train Script - Initialize Model Update (REVERTED)

**File:** `scripts/train.py`  
**Lines:** 184-211  
**Type:** Config parameter extraction  
**Reason:** Attempted to extract `use_rna_fm` from config and pass it to model constructor  
**Status:** ✅ REVERTED

**Original Code (CURRENT - after revert):**
```python
def initialize_model(config, device: str) -> CRISPRUniPredict:
    """
    Initialize model
    
    Args:
        config: Configuration object
        device: Device string
    
    Returns:
        Initialized model
    """
    model = CRISPRUniPredict(device=device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model initialized")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model
```

**Modified Code - Attempt 1 (REMOVED):**
```python
def initialize_model(config, device: str) -> CRISPRUniPredict:
    """
    Initialize model
    
    Args:
        config: Configuration object
        device: Device string
    
    Returns:
        Initialized model
    """
    # Extract model configuration
    model_config = config.get('model', {})  # <-- ADDED
    encoding_config = model_config.get('encoding', {})  # <-- ADDED
    
    # Extract use_rna_fm flag from config
    use_rna_fm = encoding_config.get('use_rna_fm', True)  # <-- ADDED
    
    model = CRISPRUniPredict(device=device, use_rna_fm=use_rna_fm)  # <-- MODIFIED
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model initialized")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  RNA-FM enabled: {use_rna_fm}")  # <-- ADDED
    
    return model
```

**Modified Code - Attempt 2 (REMOVED):**
```python
def initialize_model(config, device: str) -> CRISPRUniPredict:
    """
    Initialize model
    
    Args:
        config: Configuration object
        device: Device string
    
    Returns:
        Initialized model
    """
    # Extract use_rna_fm flag from config (via encoding config)
    use_rna_fm = config.encoding.use_rna_fm if hasattr(config, 'encoding') else True  # <-- MODIFIED APPROACH
    
    model = CRISPRUniPredict(device=device, use_rna_fm=use_rna_fm)  # <-- MODIFIED
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model initialized")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  RNA-FM enabled: {use_rna_fm}")  # <-- ADDED
    
    return model
```

---

## Undo Instructions

If you need to undo any of these changes, all files have already been reverted to their original working state as of the end of this session.

**To verify current state matches original:**
1. `utils/preprocessing/dataloader_fast.py` - Uses original `apply(is_valid)` approach
2. `models/crispr_unipredict.py` - Constructor has no `use_rna_fm` parameter
3. `scripts/train.py` - `initialize_model()` creates model with no parameters

---

## Session Notes

- **Issue:** Training was interrupted by monitoring commands (keyboard interrupt)
- **Root Cause:** User ran monitoring command which caused Ctrl+C signal to training process
- **Solution:** Undid all experimental changes and restarted training in background without monitoring
- **Final Status:** Training (Approach 1 - fast validation) now running uninterrupted in background

**Terminal ID:** `ed609957-371f-462b-bcea-0f81063c64aa`

---

## Files Modified During This Session

1. ✅ `utils/preprocessing/dataloader_fast.py` - REVERTED
2. ✅ `models/crispr_unipredict.py` - REVERTED (3 separate edits)
3. ✅ `scripts/train.py` - REVERTED (2 attempts)

All files are now in their original working state.
