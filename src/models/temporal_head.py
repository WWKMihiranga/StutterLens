"""
Temporal Detection Head.

A bidirectional LSTM with LayerNorm followed by a multi-layer classifier that
produces frame-level class logits from Wav2Vec2 embeddings.
BiLSTM offers better long-range dependency modelling than BiGRU thanks to its
explicit cell state, which helps capture prolonged stuttering events.
Deeper architecture with residual connection improves class-discriminative
temporal features.
"""

import torch
import torch.nn as nn


class TemporalDetectionHead(nn.Module):
    """BiLSTM-based frame-level classifier with LayerNorm and residual MLP."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 num_layers: int = 2, num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Hidden block: 512 → 256 → 128
        self.hidden_block = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Residual projection: match LSTM output (512) to hidden block output (128)
        self.residual_proj = nn.Linear(hidden_dim * 2, hidden_dim // 2)

        # Output head: 128 → num_classes
        self.output_head = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, features: torch.Tensor):
        """
        Parameters
        ----------
        features : (B, T, D)

        Returns
        -------
        logits         : (B, T, C)
        lstm_features  : (B, T, 2*H)   — used by the gating network
        """
        lstm_out, _ = self.lstm(features)
        lstm_out = self.layer_norm(lstm_out)

        # MLP with residual connection for better gradient flow
        hidden = self.hidden_block(lstm_out)
        residual = self.residual_proj(lstm_out)
        logits = self.output_head(hidden + residual)

        return logits, lstm_out
