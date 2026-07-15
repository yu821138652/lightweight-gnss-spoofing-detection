"""Lightweight per-signal baselines for GNSS spoofing detection."""

from __future__ import annotations

import torch
from torch import nn


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
