"""
Dedicated Off-Target CRISPR Prediction Model

Paired (sgRNA, target) input — the joint multitask model only saw sgRNA, which
is why it failed (AUROC 0.7288 vs CCLMoff 0.82). Off-target prediction needs to
see BOTH sequences to compute the mismatch pattern.

Encoding (9 channels per position):
  - 4ch sgRNA one-hot
  - 4ch target one-hot
  - 1ch mismatch indicator (1 if bases differ)

Architecture:
  - Wide multi-scale CNN (kernels 3,5,7,9)
  - Transformer encoder (global context across mismatch pattern)
  - BiGRU on label-encoded pair (concat of sgRNA+target embeddings)
  - Cross-attention fusion
  - MLP classification head (returns logits — pair with BCEWithLogits or focal loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 0}


def encode_pair(sgrnas: List[str], targets: List[str], seq_len: int = 23,
                device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a batch of (sgRNA, target) pairs.

    Returns:
        pair_onehot : (B, L, 9)  — 4ch sgRNA + 4ch target + 1ch mismatch
        pair_label  : (B, L, 2)  — integer codes for sgRNA / target (0..3),
                                   suitable for embedding lookup
    """
    B = len(sgrnas)
    pair_oh = torch.zeros(B, seq_len, 9, dtype=torch.float32)
    pair_lb = torch.zeros(B, seq_len, 2, dtype=torch.long)

    for i, (sg, tg) in enumerate(zip(sgrnas, targets)):
        sg = sg.upper().ljust(seq_len, 'N')[:seq_len]
        tg = tg.upper().ljust(seq_len, 'N')[:seq_len]
        for j in range(seq_len):
            sg_idx = NUC_TO_IDX.get(sg[j], 0)
            tg_idx = NUC_TO_IDX.get(tg[j], 0)
            pair_oh[i, j, sg_idx] = 1.0
            pair_oh[i, j, 4 + tg_idx] = 1.0
            pair_oh[i, j, 8] = 1.0 if sg[j] != tg[j] and sg[j] != 'N' and tg[j] != 'N' else 0.0
            pair_lb[i, j, 0] = sg_idx
            pair_lb[i, j, 1] = tg_idx

    return pair_oh.to(device), pair_lb.to(device)


class WideMultiScaleCNN(nn.Module):
    """Parallel conv branches over the 9-channel pair encoding."""

    KERNELS = [3, 5, 7, 9]

    def __init__(self, in_channels: int = 9, branch_channels: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in self.KERNELS:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.out_channels = branch_channels * len(self.KERNELS)
        self.residual_proj = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 9) → transpose → (B, 9, L)
        x_t = x.transpose(1, 2)
        feats = [b(x_t) for b in self.branches]
        out = torch.cat(feats, dim=1) + self.residual_proj(x_t)
        return out.transpose(1, 2)  # (B, L, out_channels)


class CrossAttentionFusion(nn.Module):
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

    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(branches, dim=1)             # (B, N, dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.ff_norm(x + self.ff(x))
        return x.mean(dim=1)


class OffTargetDedicatedModel(nn.Module):
    """
    Output: logit (B, 1). Pair with BCEWithLogitsLoss or focal loss.
    """

    def __init__(
        self,
        seq_len: int = 23,
        branch_channels: int = 64,
        embed_dim: int = 64,                # per-nucleotide embedding dim
        bigru_hidden: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.2,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len

        # ── Branch A: Wide CNN + Transformer over (B, L, 9) ────────────────
        self.cnn = WideMultiScaleCNN(in_channels=9, branch_channels=branch_channels,
                                     dropout=dropout)
        cnn_out = self.cnn.out_channels   # 64 * 4 = 256

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out, nhead=num_attention_heads,
            dim_feedforward=cnn_out * 2,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.branch_a_proj = nn.Linear(cnn_out, fusion_dim)

        # ── Branch B: Embedding + BiGRU over pair label encoding ───────────
        # Concat of sgRNA embedding and target embedding per position
        self.sg_embed = nn.Embedding(4, embed_dim)
        self.tg_embed = nn.Embedding(4, embed_dim)
        self.mismatch_embed = nn.Embedding(2, embed_dim // 2)   # 0=match, 1=mismatch

        bigru_in = embed_dim * 2 + embed_dim // 2
        self.bigru = nn.GRU(
            bigru_in, bigru_hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        self.branch_b_proj = nn.Linear(bigru_hidden * 2, fusion_dim)

        # ── Branch C: Global mismatch summary (handcrafted features) ───────
        # mismatch count per position-bin, total mismatches, GC content of sgRNA and target
        # Cheap but informative for the model.
        self.branch_c_proj = nn.Sequential(
            nn.Linear(seq_len + 5, fusion_dim),  # L mismatch flags + 5 summary feats
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = CrossAttentionFusion(dim=fusion_dim, num_heads=num_attention_heads,
                                           dropout=dropout * 0.5)

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),    # logit, NO sigmoid
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def _global_features(self, pair_onehot: torch.Tensor) -> torch.Tensor:
        """
        Cheap handcrafted summary:
          - per-position mismatch flag (seq_len floats)
          - total mismatches (1)
          - sgRNA GC content (1)
          - target GC content (1)
          - mismatches in seed region positions 0..10 (1)
          - mismatches in PAM-proximal region 11..22 (1)
        """
        mism = pair_onehot[..., 8]                          # (B, L)
        total = mism.sum(dim=1, keepdim=True)                # (B, 1)

        sg_gc = (pair_onehot[..., 1] + pair_onehot[..., 2]).sum(dim=1, keepdim=True) / self.seq_len
        tg_gc = (pair_onehot[..., 5] + pair_onehot[..., 6]).sum(dim=1, keepdim=True) / self.seq_len

        seed = mism[:, :11].sum(dim=1, keepdim=True)
        pam = mism[:, 11:].sum(dim=1, keepdim=True)

        return torch.cat([mism, total, sg_gc, tg_gc, seed, pam], dim=1)   # (B, L+5)

    def forward(self, pair_onehot: torch.Tensor, pair_label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_onehot : (B, L, 9)   from encode_pair
            pair_label  : (B, L, 2)   integer codes for sgRNA / target
        Returns:
            (B, 1) logits — use BCEWithLogitsLoss or focal loss
        """
        # Branch A
        a = self.cnn(pair_onehot)              # (B, L, 256)
        a = self.transformer(a)
        a = self._pool(a)
        a = self.branch_a_proj(a)              # (B, fusion_dim)

        # Branch B
        sg_e = self.sg_embed(pair_label[..., 0])           # (B, L, embed_dim)
        tg_e = self.tg_embed(pair_label[..., 1])           # (B, L, embed_dim)
        ms_e = self.mismatch_embed(pair_onehot[..., 8].long())  # (B, L, embed_dim/2)
        b_in = torch.cat([sg_e, tg_e, ms_e], dim=-1)       # (B, L, bigru_in)
        b_out, _ = self.bigru(b_in)                         # (B, L, hidden*2)
        b = self._pool(b_out)
        b = self.branch_b_proj(b)

        # Branch C
        c_feat = self._global_features(pair_onehot)
        c = self.branch_c_proj(c_feat)

        fused = self.fusion([a, b, c])
        return self.head(fused)                # (B, 1) logits

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ---------------------------------------------------------------------------
# Focal loss for severe class imbalance (93:1 neg:pos in CCLMoff)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss with α/γ. Operates on logits.
        FL = -α (1-p)^γ log(p)   for positives
        FL = -(1-α) p^γ log(1-p) for negatives
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)             # p if t=1 else 1-p
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


if __name__ == '__main__':
    model = OffTargetDedicatedModel()
    print('Params:', model.count_params())

    sgrnas = ['GGGTGGGGGGAGTTTGCTCCTGG'] * 4
    targets = ['GGATGGAGGGAGTTTGCTCCTGG', 'GGGTGGGGGGAGTTTGCTCCTGG',
               'TAGTGGAGGGAGCTTGCTCCTGG', 'GGGGAGGGGAAGTTTGCTCCTGG']
    pair_oh, pair_lb = encode_pair(sgrnas, targets, seq_len=23)
    print('pair_oh:', pair_oh.shape, 'pair_lb:', pair_lb.shape)

    out = model(pair_oh, pair_lb)
    print('logits:', out.shape, 'values:', out.view(-1).tolist())

    # Loss
    targets_t = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    loss = FocalLoss()(out, targets_t)
    print('focal loss:', loss.item())
