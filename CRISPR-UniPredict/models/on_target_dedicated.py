"""
Dedicated On-Target CRISPR Efficiency Prediction Model

No multitask interference — 100% focused on regression.
Architecture improvements over the base CRISPRUniPredict:
  - Wider multi-scale CNN (6 kernel sizes: 1,3,5,7,9,11)
  - Deeper BiGRU (3 layers)
  - Cross-attention fusion instead of weighted average
  - Deep regression head with BatchNorm + GELU + residuals
  - Lower dropout (0.15) tuned for regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

from models.rna_fm_encoder import RNAFMEncoder
from models.encoding import SequenceEncoder


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class WideMultiScaleCNN(nn.Module):
    """Six parallel conv branches (k=1,3,5,7,9,11) → concat → BN → residual."""

    KERNELS = [1, 3, 5, 7, 9, 11]

    def __init__(self, in_channels: int = 4, branch_channels: int = 64, dropout: float = 0.15):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in self.KERNELS:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        out_ch = branch_channels * len(self.KERNELS)   # 64 * 6 = 384
        self.residual_proj = nn.Conv1d(in_channels, out_ch, kernel_size=1)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 4)  →  transpose to (B, 4, seq_len) for Conv1d
        x_t = x.transpose(1, 2)
        feats = [b(x_t) for b in self.branches]          # each (B, 64, seq_len)
        concat = torch.cat(feats, dim=1)                  # (B, 384, seq_len)
        residual = self.residual_proj(x_t)                # (B, 384, seq_len)
        return (concat + residual).transpose(1, 2)        # (B, seq_len, 384)


class DeepBiGRU(nn.Module):
    """3-layer bidirectional GRU with layer-norm and residuals."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2   # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)   # (B, seq_len, hidden*2)
        return out


class CrossAttentionFusion(nn.Module):
    """
    Fuse N branch feature vectors via multi-head cross-attention.
    Each branch acts as a query attending to the others.
    """

    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, branches: list) -> torch.Tensor:
        # branches: list of (B, dim) tensors
        x = torch.stack(branches, dim=1)          # (B, N, dim)
        attn_out, _ = self.attn(x, x, x)         # (B, N, dim)
        x = self.norm(x + attn_out)
        x = self.ff_norm(x + self.ff(x))
        return x.mean(dim=1)                      # (B, dim) — pool over branches


class DeepRegressionHead(nn.Module):
    """
    Deep regression head with residual connections.
    256 → 256 → 128 → 64 → 32 → 1
    """

    def __init__(self, in_dim: int = 256, dropout: float = 0.15):
        super().__init__()

        def block(in_f, out_f, drop):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.GELU(),
                nn.Dropout(drop),
            )

        self.b1 = block(in_dim, 256, dropout)
        self.b2 = block(256, 128, dropout)
        self.b3 = block(128, 64, dropout * 0.8)
        self.b4 = block(64, 32, 0.0)
        self.out = nn.Linear(32, 1)

        # residual shortcuts
        self.skip1 = nn.Linear(in_dim, 256, bias=False)
        self.skip2 = nn.Linear(256, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.b1(x) + self.skip1(x)
        h2 = self.b2(h1) + self.skip2(h1)
        h3 = self.b3(h2)
        h4 = self.b4(h3)
        return self.out(h4)   # (B, 1) — unbounded; use per-source normalization


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class OnTargetDedicatedModel(nn.Module):
    """
    Dedicated model for on-target CRISPR efficiency prediction.

    Three branches:
      A) WideMultiScaleCNN + Transformer encoder
      B) Embedding + DeepBiGRU
      C) RNA-FM pretrained encoder (or learned fallback)

    Fused via CrossAttentionFusion → DeepRegressionHead.

    Labels are expected to be per-dataset z-score normalized before training
    (handled in the training script). The model therefore outputs unbounded
    real values; predictions are de-normalized for evaluation.
    """

    def __init__(
        self,
        seq_len: int = 23,
        branch_channels: int = 64,     # per-kernel channels in CNN
        bigru_hidden: int = 256,
        embed_dim: int = 128,
        vocab_size: int = 6,
        fusion_dim: int = 256,
        dropout: float = 0.15,
        rna_fm_path: str = 'models/pretrained/rna_fm_t12.pt',
        device: str = 'cpu',
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        use_rna_fm: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.device = device
        self.fusion_dim = fusion_dim

        # ── Branch A: Wide CNN + Transformer ────────────────────────────────
        self.cnn = WideMultiScaleCNN(
            in_channels=4, branch_channels=branch_channels, dropout=dropout
        )
        cnn_out = self.cnn.out_channels   # 384

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out, nhead=8, dim_feedforward=cnn_out * 2,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.branch_a_proj = nn.Linear(cnn_out, fusion_dim)

        # ── Branch B: Embedding + DeepBiGRU ─────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = DeepBiGRU(
            input_dim=embed_dim, hidden_dim=bigru_hidden,
            num_layers=3, dropout=dropout,
        )
        self.branch_b_proj = nn.Linear(self.bigru.output_dim, fusion_dim)

        # ── Branch C: RNA-FM ─────────────────────────────────────────────────
        if not use_rna_fm:
            self.rna_fm_ok = False
        else:
            try:
                self.rna_fm = RNAFMEncoder(
                    model_path=rna_fm_path,
                    freeze_layers=True,
                    num_unfreeze_layers=0,
                    device=device,
                )
                self.rna_fm_ok = True
                self.branch_c_proj = nn.Linear(640, fusion_dim)
            except Exception as e:
                warnings.warn(f"RNA-FM unavailable ({e}). Using learned fallback.")
                self.rna_fm_ok = False

        if not self.rna_fm_ok:
            self.branch_c_fallback = nn.Sequential(
                nn.Linear(embed_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Linear(fusion_dim * 2, fusion_dim),
            )

        # ── Fusion ───────────────────────────────────────────────────────────
        self.fusion = CrossAttentionFusion(
            dim=fusion_dim, num_heads=num_attention_heads, dropout=dropout * 0.5,
        )

        # ── Regression head ──────────────────────────────────────────────────
        self.head = DeepRegressionHead(in_dim=fusion_dim, dropout=dropout)

        self.encoder = SequenceEncoder(device=device)
        self.to(device)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Global average pool over sequence dim: (B, L, D) → (B, D)."""
        return x.mean(dim=1)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        sgrna_onehot: torch.Tensor,
        sgrna_label: torch.Tensor,
        rna_emb: Optional[torch.Tensor] = None,
        sgrna_strs: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Args:
            sgrna_onehot : (B, seq_len, 4)  one-hot encoded
            sgrna_label  : (B, seq_len)     integer encoded
            rna_emb      : (B, 640) pre-computed RNA-FM embeddings (preferred)
            sgrna_strs   : raw string list for RNA-FM (fallback, slow)
        Returns:
            (B, 1) predicted efficiency (z-score space)
        """
        # Branch A
        a = self.cnn(sgrna_onehot)             # (B, L, 384)
        a = self.transformer(a)                # (B, L, 384)
        a = self._pool(a)                      # (B, 384)
        a = self.branch_a_proj(a)              # (B, 256)

        # Branch B
        b = self.embedding(sgrna_label)        # (B, L, 128)
        b = self.bigru(b)                      # (B, L, 512)
        b = self._pool(b)                      # (B, 512)
        b = self.branch_b_proj(b)              # (B, 256)

        # Branch C — use pre-computed embedding if available and RNA-FM enabled
        if rna_emb is not None and self.rna_fm_ok:
            c = self.branch_c_proj(rna_emb.to(dtype=torch.float32))   # (B, 256)
        elif self.rna_fm_ok and sgrna_strs is not None:
            with torch.no_grad():
                c = self.rna_fm.encode_batch_pairs(sgrna_strs, sgrna_strs)
            c = self.branch_c_proj(c.to(device=sgrna_onehot.device, dtype=torch.float32))
        elif self.rna_fm_ok:
            emb = self.embedding(sgrna_label).mean(dim=1)
            pad = torch.zeros(emb.size(0), 640 - emb.size(1), device=emb.device)
            c   = self.branch_c_proj(torch.cat([emb, pad], dim=1))
        else:
            emb = self.embedding(sgrna_label).mean(dim=1)
            c   = self.branch_c_fallback(emb)  # (B, 256)

        # Fusion + head
        fused = self.fusion([a, b, c])         # (B, 256)
        return self.head(fused)                # (B, 1)

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
