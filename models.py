"""
Sparse Attention Transformer Models
====================================
Three attention mechanisms implemented from scratch:
1. Full Attention - Standard O(n^2) attention (baseline)
2. Random Sparse Attention - Randomly samples tokens
3. Top-k Sparse Attention - MiniMax-style, attends only to top-k scoring tokens

All run on CPU. No GPU required.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class FullAttention(nn.Module):
    """Standard scaled dot-product attention. O(n^2) complexity."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len) or None
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        B, L, _ = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.W_o(output)

        return output, attn_weights


class RandomSparseAttention(nn.Module):
    """
    Random sparse attention: each query attends to a random subset of keys.
    Useful as a baseline to show that random selection != intelligent selection.
    """

    def __init__(self, d_model: int, n_heads: int, k: int = 20, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.k = k

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        k = min(self.k, L)

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Full scores for visualization, but we'll mask out non-selected
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Random selection mask: for each query, randomly pick k keys
        rand_mask = torch.zeros(B, self.n_heads, L, L, device=x.device, dtype=torch.bool)
        for b in range(B):
            for h in range(self.n_heads):
                for i in range(L):
                    indices = torch.randperm(L, device=x.device)[:k]
                    rand_mask[b, h, i, indices] = True

        # Apply random mask
        scores = scores.masked_fill(~rand_mask, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.W_o(output)

        return output, attn_weights


class TopKSparseAttention(nn.Module):
    """
    Top-k Sparse Attention (MiniMax-style).

    For each query token, compute full attention scores, then keep only
    the top-k highest scoring keys. Everything else is masked out.

    This is the core idea behind MiniMax's sparse attention that enables
    processing 1M+ tokens efficiently.
    """

    def __init__(self, d_model: int, n_heads: int, k: int = 20, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.k = k

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        k = min(self.k, L)

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Compute full attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Top-k selection: keep only the k highest scores per query
        topk_values, topk_indices = torch.topk(scores, k=k, dim=-1)

        # Build sparse mask
        sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
        sparse_mask.scatter_(-1, topk_indices, True)

        # Zero out non-top-k positions
        scores = scores.masked_fill(~sparse_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.W_o(output)

        return output, attn_weights


class TransformerBlock(nn.Module):
    """Single transformer block: attention + FFN + LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attention_type: str = "full",
        k: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        if attention_type == "full":
            self.attention = FullAttention(d_model, n_heads, dropout)
        elif attention_type == "random":
            self.attention = RandomSparseAttention(d_model, n_heads, k, dropout)
        elif attention_type == "topk":
            self.attention = TopKSparseAttention(d_model, n_heads, k, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, mask)
        x = x + attn_out

        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x, attn_weights


class SparseAttentionClassifier(nn.Module):
    """
    Text classifier using configurable attention mechanism.

    Architecture:
    - Embedding (vocab -> d_model)
    - Positional encoding (sinusoidal)
    - N transformer blocks
    - Mean pooling
    - Classification head
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        n_classes: int = 2,
        max_seq_len: int = 512,
        attention_type: str = "full",
        k: int = 20,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = self._build_positional_encoding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, attention_type, k, dropout)
                for _ in range(n_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_classes),
        )

        self._init_weights()

    def _build_positional_encoding(
        self, max_len: int, d_model: int
    ) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, max_len, d_model)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) 1=real, 0=pad
        Returns:
            dict with 'logits', 'attention_weights' (list per layer)
        """
        B, L = input_ids.shape

        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        pe_len = self.pos_encoding.size(1)
        if L > pe_len:
            x = x[:, :pe_len, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :pe_len]
            L = pe_len
        x = x + self.pos_encoding[:, :L, :]
        x = self.embed_dropout(x)

        # Build mask for attention: (B, 1, 1, L)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        else:
            mask = None

        # Transformer blocks
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x, mask)
            all_attn_weights.append(attn_w.detach())

        # Mean pooling (excluding padding)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        logits = self.classifier(x)

        return {"logits": logits, "attention_weights": all_attn_weights}


def build_model(
    vocab_size: int,
    attention_type: str = "full",
    k: int = 20,
    n_classes: int = 2,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    max_seq_len: int = 256,
) -> SparseAttentionClassifier:
    """Factory function to build a model with given attention type."""
    return SparseAttentionClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 2,
        n_classes=n_classes,
        max_seq_len=max_seq_len,
        attention_type=attention_type,
        k=k,
    )
