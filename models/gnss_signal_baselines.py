"""Lightweight per-signal baselines for GNSS spoofing detection."""

from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Module):
    """One-dimensional convolution whose output at t only sees samples up to t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.left_padding, 0)))


class SignalMLP(nn.Module):
    """Classify each signal by flattening its short causal feature window."""

    def __init__(self, input_dim: int, time_steps: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim * time_steps),
            nn.Linear(input_dim * time_steps, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-signal binary logits for ``[batch, signal, time, feature]`` input."""
        batch_size, signal_count, time_steps, input_dim = x.shape
        if time_steps != self.time_steps or input_dim != self.input_dim:
            raise ValueError(f"Expected [*, *, {self.time_steps}, {self.input_dim}], got {tuple(x.shape)}")
        flattened = x.reshape(batch_size * signal_count, time_steps * input_dim)
        return self.classifier(flattened).reshape(batch_size, signal_count, 2)


class SignalGRU(nn.Module):
    """Classify each signal with a small recurrent encoder over its time window."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-signal binary logits for ``[batch, signal, time, feature]`` input."""
        batch_size, signal_count, _, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {input_dim}")
        sequence = x.reshape(batch_size * signal_count, x.shape[2], input_dim)
        _, hidden = self.gru(sequence)
        return self.classifier(hidden[-1]).reshape(batch_size, signal_count, 2)


class SignalTCN(nn.Module):
    """Classify each signal with a compact causal temporal convolutional encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            CausalConv1d(input_dim, hidden_dim, kernel_size=3),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-signal logits without using observations after the current epoch."""
        batch_size, signal_count, _, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {input_dim}")
        sequence = x.reshape(batch_size * signal_count, x.shape[2], input_dim).transpose(1, 2)
        encoded = self.encoder(sequence)
        return self.classifier(encoded[:, :, -1]).reshape(batch_size, signal_count, 2)


class SignalLSTM(nn.Module):
    """Classify each signal with a small LSTM over the causal feature window."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-signal binary logits for ``[batch, signal, time, feature]`` input."""
        batch_size, signal_count, _, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {input_dim}")
        sequence = x.reshape(batch_size * signal_count, x.shape[2], input_dim)
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(hidden[-1]).reshape(batch_size, signal_count, 2)


class SignalTransformerTiny(nn.Module):
    """One-layer causal Transformer kept small enough for edge-model comparison."""

    def __init__(
        self,
        input_dim: int,
        time_steps: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, time_steps, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits from the current token under an upper-triangular causal mask."""
        batch_size, signal_count, time_steps, input_dim = x.shape
        if time_steps != self.time_steps or input_dim != self.input_dim:
            raise ValueError(f"Expected [*, *, {self.time_steps}, {self.input_dim}], got {tuple(x.shape)}")
        sequence = x.reshape(batch_size * signal_count, time_steps, input_dim)
        sequence = self.input_projection(sequence) + self.position_embedding
        causal_mask = torch.triu(
            torch.ones(time_steps, time_steps, dtype=torch.bool, device=x.device), diagonal=1
        )
        encoded = self.encoder(sequence, mask=causal_mask)
        return self.classifier(encoded[:, -1]).reshape(batch_size, signal_count, 2)
