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


class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise temporal convolution for efficient edge models."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, groups=channels
        )

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


class SignalRawStatsFusion(nn.Module):
    """Fuse a true raw temporal encoder with an MLP statistics branch."""

    def __init__(self, raw_input_dim: int, stats_input_dim: int, encoder: str = "lstm", hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.stats_input_dim = stats_input_dim
        self.encoder_name = encoder
        if encoder == "lstm":
            self.raw_encoder = nn.LSTM(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "tcn":
            self.raw_encoder = nn.Sequential(
                CausalConv1d(raw_input_dim, hidden_dim, kernel_size=3), nn.GELU(), nn.Dropout(dropout),
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2), nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown raw encoder: {encoder}")
        self.stats_encoder = nn.Sequential(
            nn.LayerNorm(stats_input_dim), nn.Linear(stats_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )

    def forward(self, raw_x: torch.Tensor, stats_x: torch.Tensor) -> torch.Tensor:
        batch_size, signal_count, _, raw_dim = raw_x.shape
        if raw_dim != self.raw_input_dim or stats_x.shape[-1] != self.stats_input_dim or stats_x.shape[-2] != 1:
            raise ValueError(f"Unexpected fusion inputs: raw={tuple(raw_x.shape)} stats={tuple(stats_x.shape)}")
        raw = raw_x.reshape(batch_size * signal_count, raw_x.shape[2], raw_dim)
        if self.encoder_name == "lstm":
            _, (hidden, _) = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        else:
            raw_embedding = self.raw_encoder(raw.transpose(1, 2))[:, :, -1]
        stats = stats_x.reshape(batch_size * signal_count, self.stats_input_dim)
        fused = torch.cat([raw_embedding, self.stats_encoder(stats)], dim=-1)
        return self.classifier(fused).reshape(batch_size, signal_count, 2)


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


class DeviceStatsMLP(nn.Module):
    """Lowest-complexity device alarm baseline over a causal statistics window."""

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
        if x.ndim != 3 or tuple(x.shape[-2:]) != (self.time_steps, self.input_dim):
            raise ValueError(f"Expected [batch, {self.time_steps}, {self.input_dim}], got {tuple(x.shape)}")
        return self.classifier(x.reshape(x.shape[0], -1))


class DeviceStatsGRU(nn.Module):
    """Small GRU that emits one spoofing alarm for an aggregated device window."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch, time, {self.input_dim}], got {tuple(x.shape)}")
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])


class DeviceStatsLSTM(nn.Module):
    """Small LSTM alternative for direct device-level alarm prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch, time, {self.input_dim}], got {tuple(x.shape)}")
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


class DeviceStatsTCN(nn.Module):
    """Causal convolutional device-level baseline for short GNSS windows."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, dropout: float = 0.1):
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
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch, time, {self.input_dim}], got {tuple(x.shape)}")
        encoded = self.encoder(x.transpose(1, 2))
        return self.classifier(encoded[:, :, -1])


class DeviceStatsDepthwiseCNN(nn.Module):
    """Depthwise-separable causal CNN for direct edge-device alarms."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.stem = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.depthwise_short = CausalDepthwiseConv1d(hidden_dim, kernel_size=3)
        self.depthwise_long = CausalDepthwiseConv1d(hidden_dim, kernel_size=3, dilation=2)
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch, time, {self.input_dim}], got {tuple(x.shape)}")
        encoded = self.stem(x.transpose(1, 2))
        encoded = encoded + self.depthwise_short(encoded)
        encoded = nn.functional.gelu(encoded)
        encoded = encoded + self.depthwise_long(encoded)
        encoded = nn.functional.gelu(self.pointwise(encoded))
        return self.classifier(encoded[:, :, -1])


class DeviceStatsNLinear(nn.Module):
    """NLinear-inspired classifier using deviations from the current baseline."""

    def __init__(self, input_dim: int, time_steps: int, hidden_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.time_projection = nn.Linear(time_steps, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[-2:]) != (self.time_steps, self.input_dim):
            raise ValueError(f"Expected [batch, {self.time_steps}, {self.input_dim}], got {tuple(x.shape)}")
        deviations = x - x[:, -1:, :].detach()
        compressed = self.time_projection(deviations.transpose(1, 2)).squeeze(-1)
        return self.classifier(compressed)


class DeviceStatsDLinear(nn.Module):
    """DLinear-inspired classifier with explicit trend and residual projections."""

    def __init__(self, input_dim: int, time_steps: int, hidden_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.kernel_size = min(5, time_steps if time_steps % 2 else time_steps - 1)
        self.seasonal_projection = nn.Linear(time_steps, 1)
        self.trend_projection = nn.Linear(time_steps, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def moving_average(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        padded = nn.functional.pad(x.transpose(1, 2), (padding, padding), mode="replicate")
        return nn.functional.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[-2:]) != (self.time_steps, self.input_dim):
            raise ValueError(f"Expected [batch, {self.time_steps}, {self.input_dim}], got {tuple(x.shape)}")
        trend = self.moving_average(x)
        seasonal = x - trend
        trend_features = self.trend_projection(trend.transpose(1, 2)).squeeze(-1)
        seasonal_features = self.seasonal_projection(seasonal.transpose(1, 2)).squeeze(-1)
        return self.classifier(torch.cat([trend_features, seasonal_features], dim=-1))


class DeviceStatsTSMixer(nn.Module):
    """Compact TSMixer-style classifier over a complete causal device window."""

    def __init__(
        self,
        input_dim: int,
        time_steps: int,
        hidden_dim: int = 24,
        dropout: float = 0.1,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.time_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_blocks)])
        self.time_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(time_steps, time_steps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(time_steps, time_steps),
                nn.Dropout(dropout),
            )
            for _ in range(num_blocks)
        ])
        self.feature_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_blocks)])
        self.feature_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[-2:]) != (self.time_steps, self.input_dim):
            raise ValueError(f"Expected [batch, {self.time_steps}, {self.input_dim}], got {tuple(x.shape)}")
        for time_norm, time_mixer, feature_norm, feature_mixer in zip(
            self.time_norms, self.time_mixers, self.feature_norms, self.feature_mixers
        ):
            x = x + time_mixer(time_norm(x).transpose(1, 2)).transpose(1, 2)
            x = x + feature_mixer(feature_norm(x))
        return self.classifier(x[:, -1])
