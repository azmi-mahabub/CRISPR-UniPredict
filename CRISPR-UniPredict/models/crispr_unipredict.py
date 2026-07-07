"""
CRISPR-UniPredict: Unified Hybrid Architecture for CRISPR Prediction
Combines three complementary branches:
- Branch A: One-hot encoding → MSC → MHSA (local sequence patterns + attention)
- Branch B: Label encoding → Embedding → BiGRU (sequential context)
- Branch C: Label encoding → RNA-FM encoder (pretrained contextual embeddings)

Multi-task learning for on-target and off-target prediction
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union
from pathlib import Path
import warnings

from models.msc_module import MultiScaleConvolution
from models.mhsa_module import MultiHeadSelfAttention
from models.bigru_module import BiGRUModule
from models.rna_fm_encoder import RNAFMEncoder
from models.encoding import SequenceEncoder


class AttentionFusion(nn.Module):
    """
    Attention-based feature fusion layer
    Learns to weight and combine features from multiple branches
    """
    
    def __init__(self, feature_dim: int, num_branches: int = 3):
        """
        Initialize attention fusion layer
        
        Args:
            feature_dim: Dimension of input features
            num_branches: Number of branches to fuse
        """
        super(AttentionFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_branches = num_branches
        
        # Attention weights for each branch
        self.branch_attention = nn.Sequential(
            nn.Linear(feature_dim * num_branches, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_branches),
            nn.Softmax(dim=-1)
        )
        
        # Fusion projection layer
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim * num_branches, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.35)
        )
    
    def forward(self, branch_features: list) -> torch.Tensor:
        """
        Fuse features from multiple branches using attention
        
        Args:
            branch_features: List of tensors from different branches
                           Each of shape (batch, feature_dim)
        
        Returns:
            Fused features of shape (batch, feature_dim)
        """
        # Concatenate all branch features
        concatenated = torch.cat(branch_features, dim=-1)  # (batch, feature_dim * num_branches)
        
        # Compute attention weights for each branch
        attention_weights = self.branch_attention(concatenated)  # (batch, num_branches)
        
        # Apply attention weights to each branch
        weighted_features = []
        for i, features in enumerate(branch_features):
            weight = attention_weights[:, i:i+1]  # (batch, 1)
            weighted_features.append(features * weight)
        
        # Concatenate weighted features
        weighted_concat = torch.cat(weighted_features, dim=-1)
        
        # Project to final feature dimension
        fused = self.fusion_proj(weighted_concat)
        
        return fused


class CRISPRUniPredict(nn.Module):
    """
    CRISPR-UniPredict: Unified Hybrid Neural Network for CRISPR Prediction
    
    Multi-branch architecture combining:
    1. CNN-based local pattern recognition (MSC + MHSA)
    2. Sequential context modeling (BiGRU)
    3. Pretrained language model embeddings (RNA-FM)
    
    Supports multi-task learning for on-target and off-target prediction
    
    Architecture:
    - Branch A: One-hot → MSC (256) → MHSA → GlobalAvgPool
    - Branch B: Label → Embedding (128) → BiGRU (128) → GlobalAvgPool
    - Branch C: Label → RNA-FM (640) → GlobalAvgPool
    - Fusion: Attention-based fusion of all branches
    - Task Heads: Separate heads for on-target and off-target prediction
    """
    
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
        super(CRISPRUniPredict, self).__init__()
        
        self.seq_len = seq_len
        self.device = device
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # ============ BRANCH A: One-hot → MSC → MHSA ============
        self.msc = MultiScaleConvolution(
            in_channels=4,  # One-hot encoded (A, C, G, T)
            out_channels=msc_out_channels,
            dropout=dropout
        )
        
        # MSC output channels: 64 * 4 = 256
        msc_out_dim = self.msc.get_output_channels()
        
        # Project MSC output to MHSA embedding dimension if needed
        if msc_out_dim != mhsa_embed_dim:
            self.branch_a_proj = nn.Linear(msc_out_dim, mhsa_embed_dim)
        else:
            self.branch_a_proj = None
        
        self.mhsa = MultiHeadSelfAttention(
            embed_dim=mhsa_embed_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # ============ BRANCH B: Label → Embedding → BiGRU ============
        self.label_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        self.bigru = BiGRUModule(
            input_dim=embedding_dim,
            hidden_dim=bigru_hidden_dim,
            num_layers=1,
            dropout=dropout,
            bidirectional=True
        )
        
        bigru_out_dim = self.bigru.output_dim  # 256 (128 * 2)
        
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
        
        # ============ FEATURE FUSION ============
        # Project all branches to same dimension for fusion
        self.branch_a_pool = nn.AdaptiveAvgPool1d(1)
        self.branch_b_pool = nn.AdaptiveAvgPool1d(1)
        self.branch_c_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection layers to unified dimension
        self.branch_a_fusion_proj = nn.Linear(mhsa_embed_dim, hidden_dim)
        self.branch_b_fusion_proj = nn.Linear(bigru_out_dim, hidden_dim)
        self.branch_c_fusion_proj = nn.Linear(rna_fm_out_dim, hidden_dim)
        
        # Attention-based fusion
        self.fusion = AttentionFusion(hidden_dim, num_branches=3)
        
        # Residual connection for fusion
        self.fusion_residual_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # ============ SHARED REPRESENTATION ============
        self.shared_representation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============ TASK HEAD 1: On-target Prediction ============
        self.on_target_head = nn.Sequential(
            nn.Linear(hidden_dim, 80),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(80, 20),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(20, 1),
            nn.Sigmoid()  # Output: 0-1 score
        )
        
        # ============ TASK HEAD 2: Off-target Prediction ============
        self.off_target_head = nn.Sequential(
            nn.Linear(hidden_dim, 80),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(80, 20),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(20, 1)
            # No Sigmoid: BCEWithLogitsLoss applies sigmoid internally
        )
        
        # Sequence encoder for preprocessing
        self.encoder = SequenceEncoder(device=device)
        
        # Move model to device
        self.to(device)
    
    def encode_sequences(self, sgrna: str, target: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sgRNA and target sequences
        
        Args:
            sgrna: sgRNA sequence string
            target: Target sequence string
        
        Returns:
            Tuple of (one_hot_encoded, label_encoded) tensors
        """
        # One-hot encoding for Branch A
        one_hot = self.encoder.encode_one_hot(sgrna)  # (1, seq_len, 4)
        
        # Label encoding for Branches B and C
        label_encoded = self.encoder.encode_label(sgrna)  # (1, seq_len)
        
        return one_hot, label_encoded
    
    def forward(self, sgrna_onehot: torch.Tensor, 
                sgrna_label: torch.Tensor,
                task_type: str = 'both',
                sgrna_strs: Optional[list] = None,
                target_strs: Optional[list] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through CRISPR-UniPredict
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA of shape (batch, seq_len, 4)
            sgrna_label: Label encoded sgRNA of shape (batch, seq_len)
            task_type: Type of prediction ('on_target', 'off_target', or 'both')
        
        Returns:
            If task_type == 'on_target': on-target prediction (batch, 1)
            If task_type == 'off_target': off-target prediction (batch, 1)
            If task_type == 'both': tuple of (on_target, off_target) predictions
        """
        batch_size = sgrna_onehot.size(0)
        
        # ============ BRANCH A: One-hot → MSC → MHSA ============
        branch_a = self.msc(sgrna_onehot)  # (batch, seq_len, 256)
        
        # Project if needed
        if self.branch_a_proj is not None:
            branch_a = self.branch_a_proj(branch_a)
        
        # Apply MHSA
        branch_a = self.mhsa(branch_a)  # (batch, seq_len, 256)
        
        # Global average pooling
        branch_a = branch_a.transpose(1, 2)  # (batch, 256, seq_len)
        branch_a = self.branch_a_pool(branch_a)  # (batch, 256, 1)
        branch_a = branch_a.squeeze(-1)  # (batch, 256)
        
        # ============ BRANCH B: Label → Embedding → BiGRU ============
        branch_b = self.label_embedding(sgrna_label)  # (batch, seq_len, embedding_dim)
        branch_b = self.bigru(branch_b)  # (batch, seq_len, 256)
        
        # Global average pooling
        branch_b = branch_b.transpose(1, 2)  # (batch, 256, seq_len)
        branch_b = self.branch_b_pool(branch_b)  # (batch, 256, 1)
        branch_b = branch_b.squeeze(-1)  # (batch, 256)
        
        # ============ BRANCH C: RNA-FM (pair) or embedding fallback ============
        if self.rna_fm_available:
            if sgrna_strs is not None and target_strs is not None:
                # Use optimized batch encoding instead of sequential loop
                branch_c = self.rna_fm.encode_batch_pairs(sgrna_strs, target_strs)
                # Ensure correct device and dtype
                branch_c = branch_c.to(
                    device=sgrna_onehot.device, dtype=torch.float32
                )
            else:
                pooled = self.label_embedding(sgrna_label).mean(dim=1)
                branch_c = self.branch_c_embed_proj(pooled)
        else:
            branch_c = self.label_embedding(sgrna_label).mean(dim=1)
            branch_c = self.rna_fm_fallback(branch_c)
        
        # ============ FEATURE FUSION ============
        # Project branches to unified dimension
        branch_a_fused = self.branch_a_fusion_proj(branch_a)  # (batch, hidden_dim)
        branch_b_fused = self.branch_b_fusion_proj(branch_b)  # (batch, hidden_dim)
        branch_c_fused = self.branch_c_fusion_proj(branch_c)  # (batch, hidden_dim)
        
        # Apply attention-based fusion
        fused = self.fusion([branch_a_fused, branch_b_fused, branch_c_fused])  # (batch, hidden_dim)
        
        # Add residual connection from concatenated branches
        concatenated = torch.cat([branch_a_fused, branch_b_fused, branch_c_fused], dim=-1)
        residual = self.fusion_residual_proj(concatenated)
        fused = fused + residual  # Residual connection
        
        # ============ SHARED REPRESENTATION ============
        shared = self.shared_representation(fused)  # (batch, hidden_dim)
        
        # ============ TASK HEADS ============
        if task_type == 'on_target':
            return self.on_target_head(shared)
        elif task_type == 'off_target':
            return self.off_target_head(shared)
        elif task_type == 'both':
            on_target = self.on_target_head(shared)
            off_target = self.off_target_head(shared)
            return on_target, off_target
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Must be 'on_target', 'off_target', or 'both'")
    
    def predict_on_target(self, sgrna_onehot: torch.Tensor, 
                         sgrna_label: torch.Tensor) -> torch.Tensor:
        """
        Predict on-target activity
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA
            sgrna_label: Label encoded sgRNA
        
        Returns:
            On-target prediction score (0-1)
        """
        return self.forward(sgrna_onehot, sgrna_label, task_type='on_target')
    
    def predict_off_target(self, sgrna_onehot: torch.Tensor,
                          sgrna_label: torch.Tensor) -> torch.Tensor:
        """
        Predict off-target probability
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA
            sgrna_label: Label encoded sgRNA
        
        Returns:
            Off-target prediction probability (0-1)
        """
        return self.forward(sgrna_onehot, sgrna_label, task_type='off_target')
    
    def freeze_branch_a(self):
        """Freeze Branch A (MSC + MHSA) for selective training"""
        for param in self.msc.parameters():
            param.requires_grad = False
        for param in self.mhsa.parameters():
            param.requires_grad = False
        if self.branch_a_proj is not None:
            for param in self.branch_a_proj.parameters():
                param.requires_grad = False
    
    def freeze_branch_b(self):
        """Freeze Branch B (BiGRU) for selective training"""
        for param in self.label_embedding.parameters():
            param.requires_grad = False
        for param in self.bigru.parameters():
            param.requires_grad = False
    
    def freeze_branch_c(self):
        """Freeze Branch C (RNA-FM) for selective training"""
        if self.rna_fm_available:
            for param in self.rna_fm.parameters():
                param.requires_grad = False
            for param in self.branch_c_embed_proj.parameters():
                param.requires_grad = False
        else:
            for param in self.rna_fm_fallback.parameters():
                param.requires_grad = False
    
    def unfreeze_branch_a(self):
        """Unfreeze Branch A for training"""
        for param in self.msc.parameters():
            param.requires_grad = True
        for param in self.mhsa.parameters():
            param.requires_grad = True
        if self.branch_a_proj is not None:
            for param in self.branch_a_proj.parameters():
                param.requires_grad = True
    
    def unfreeze_branch_b(self):
        """Unfreeze Branch B for training"""
        for param in self.label_embedding.parameters():
            param.requires_grad = True
        for param in self.bigru.parameters():
            param.requires_grad = True
    
    def unfreeze_branch_c(self):
        """Unfreeze Branch C for training"""
        if self.rna_fm_available:
            for param in self.rna_fm.parameters():
                param.requires_grad = True
            for param in self.branch_c_embed_proj.parameters():
                param.requires_grad = True
        else:
            for param in self.rna_fm_fallback.parameters():
                param.requires_grad = True
    
    def get_total_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """Get model configuration and parameter information"""
        total_params = self.get_total_params()
        trainable_params = self.get_trainable_params()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'seq_len': self.seq_len,
            'device': str(self.device),
            'rna_fm_available': self.rna_fm_available,
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("CRISPR-UNIPREDICT MODEL TESTING")
    print("=" * 80)
    
    # Test 1: Model initialization
    print("\n1. MODEL INITIALIZATION")
    print("-" * 80)
    
    device = 'cpu'
    model = CRISPRUniPredict(
        seq_len=23,
        msc_out_channels=64,
        mhsa_embed_dim=256,
        bigru_hidden_dim=128,
        embedding_dim=128,
        hidden_dim=256,
        dropout=0.35,
        device=device
    )
    
    print(f"[OK] Model initialized on device: {device}")
    
    # Test 2: Model info
    print("\n2. MODEL INFORMATION")
    print("-" * 80)
    
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test 3: Forward pass - on-target prediction
    print("\n3. FORWARD PASS - ON-TARGET PREDICTION")
    print("-" * 80)
    
    batch_size = 2
    seq_len = 23
    
    # Create dummy inputs
    sgrna_onehot = torch.randn(batch_size, seq_len, 4)
    sgrna_label = torch.randint(0, 6, (batch_size, seq_len))
    
    print(f"Input shapes:")
    print(f"  sgrna_onehot: {sgrna_onehot.shape}")
    print(f"  sgrna_label: {sgrna_label.shape}")
    
    with torch.no_grad():
        on_target = model.predict_on_target(sgrna_onehot, sgrna_label)
    
    print(f"[OK] On-target prediction shape: {on_target.shape}")
    print(f"  Output range: [{on_target.min().item():.4f}, {on_target.max().item():.4f}]")
    
    # Test 4: Forward pass - off-target prediction
    print("\n4. FORWARD PASS - OFF-TARGET PREDICTION")
    print("-" * 80)
    
    with torch.no_grad():
        off_target = model.predict_off_target(sgrna_onehot, sgrna_label)
    
    print(f"[OK] Off-target prediction shape: {off_target.shape}")
    print(f"  Output range: [{off_target.min().item():.4f}, {off_target.max().item():.4f}]")
    
    # Test 5: Forward pass - both tasks
    print("\n5. FORWARD PASS - BOTH TASKS")
    print("-" * 80)
    
    with torch.no_grad():
        on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
    
    print(f"[OK] On-target shape: {on_target.shape}")
    print(f"[OK] Off-target shape: {off_target.shape}")
    
    # Test 6: Gradient flow
    print("\n6. GRADIENT FLOW TEST")
    print("-" * 80)
    
    sgrna_onehot = torch.randn(batch_size, seq_len, 4, requires_grad=True)
    sgrna_label = torch.randint(0, 6, (batch_size, seq_len))
    
    on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
    
    loss = on_target.sum() + off_target.sum()
    loss.backward()
    
    print(f"[OK] Gradients computed successfully")
    print(f"  Input gradient shape: {sgrna_onehot.grad.shape}")
    print(f"  Input gradient mean: {sgrna_onehot.grad.mean().item():.6f}")
    
    # Test 7: Selective freezing
    print("\n7. SELECTIVE BRANCH FREEZING")
    print("-" * 80)
    
    model.freeze_branch_a()
    print(f"[OK] Branch A frozen")
    
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model.unfreeze_branch_a()
    print(f"[OK] Branch A unfrozen")
    
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params before: {trainable_before:,}")
    print(f"  Trainable params after: {trainable_after:,}")
    
    # Test 8: Different batch sizes
    print("\n8. VARIABLE BATCH SIZES")
    print("-" * 80)
    
    for batch_size in [1, 4, 8]:
        sgrna_onehot = torch.randn(batch_size, seq_len, 4)
        sgrna_label = torch.randint(0, 6, (batch_size, seq_len))
        
        with torch.no_grad():
            on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
        
        print(f"  Batch size {batch_size}: on_target {on_target.shape}, off_target {off_target.shape}")
    
    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
