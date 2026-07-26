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


class MiniTimesBlock(nn.Module):
    """Small TimesNet-style period block for short causal history windows."""

    def __init__(self, hidden_dim: int, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.period_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _period_candidates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps = x.shape[1]
        spectrum = torch.fft.rfft(x, dim=1).abs().mean(dim=(0, 2))
        if spectrum.numel() <= 1:
            periods = torch.ones(1, dtype=torch.long, device=x.device)
            weights = torch.ones(1, dtype=x.dtype, device=x.device)
            return periods, weights
        spectrum = spectrum.clone()
        spectrum[0] = 0
        k = min(self.top_k, spectrum.numel() - 1)
        weights, frequency_indices = torch.topk(spectrum, k=k)
        periods = torch.div(time_steps, frequency_indices, rounding_mode="floor").clamp(min=1)
        return periods, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, hidden_dim = x.shape
        periods, weights = self._period_candidates(x)
        outputs = []
        for period in periods.tolist():
            padded_steps = ((time_steps + period - 1) // period) * period
            if padded_steps > time_steps:
                padded = nn.functional.pad(x, (0, 0, 0, padded_steps - time_steps))
            else:
                padded = x
            folded = padded.reshape(batch_size, padded_steps // period, period, hidden_dim)
            folded = folded.permute(0, 3, 1, 2).contiguous()
            encoded = self.period_conv(folded)
            encoded = encoded.permute(0, 2, 3, 1).reshape(batch_size, padded_steps, hidden_dim)
            outputs.append(encoded[:, :time_steps, :])
        stacked = torch.stack(outputs, dim=-1)
        normalized_weights = torch.softmax(weights, dim=0).to(dtype=x.dtype).view(1, 1, 1, -1)
        mixed = (stacked * normalized_weights).sum(dim=-1)
        return self.norm(x + mixed)


class MiniTimesNetEncoder(nn.Module):
    """Compact TimesNet-inspired sequence encoder returning the final token."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1, num_blocks: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MiniTimesBlock(hidden_dim=hidden_dim, top_k=2, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch, time, {self.input_dim}], got {tuple(x.shape)}")
        encoded = self.input_projection(x)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded[:, -1]


class InceptionBlockV1(nn.Module):
    """TimesNet-style multi-kernel 2D convolution block."""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        kernels = [2 * index + 1 for index in range(num_kernels)]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)
            for kernel in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [conv(x) for conv in self.convs]
        return torch.stack(outputs, dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    """Fuller TimesNet block with FFT period routing and Inception 2D kernels."""

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(hidden_dim, ff_dim, num_kernels=num_kernels),
            nn.GELU(),
            nn.Dropout(dropout),
            InceptionBlockV1(ff_dim, hidden_dim, num_kernels=num_kernels),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _period_candidates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps = x.shape[1]
        spectrum = torch.fft.rfft(x, dim=1).abs().mean(dim=(0, 2))
        if spectrum.numel() <= 1:
            periods = torch.ones(1, dtype=torch.long, device=x.device)
            weights = torch.ones(1, dtype=x.dtype, device=x.device)
            return periods, weights
        spectrum = spectrum.clone()
        spectrum[0] = 0
        k = min(self.top_k, spectrum.numel() - 1)
        weights, frequency_indices = torch.topk(spectrum, k=k)
        periods = torch.div(time_steps, frequency_indices, rounding_mode="floor").clamp(min=1)
        return periods, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, hidden_dim = x.shape
        periods, weights = self._period_candidates(x)
        outputs = []
        for period in periods.tolist():
            padded_steps = ((time_steps + period - 1) // period) * period
            if padded_steps > time_steps:
                padded = nn.functional.pad(x, (0, 0, 0, padded_steps - time_steps))
            else:
                padded = x
            folded = padded.reshape(batch_size, padded_steps // period, period, hidden_dim)
            folded = folded.permute(0, 3, 1, 2).contiguous()
            encoded = self.conv(folded)
            encoded = encoded.permute(0, 2, 3, 1).reshape(batch_size, padded_steps, hidden_dim)
            outputs.append(encoded[:, :time_steps, :])
        stacked = torch.stack(outputs, dim=-1)
        normalized_weights = torch.softmax(weights, dim=0).to(dtype=x.dtype).view(1, 1, 1, -1)
        mixed = (stacked * normalized_weights).sum(dim=-1)
        return self.norm(x + mixed)


class FullTimesNetEncoder(nn.Module):
    """TimesNet-paper-style encoder scaled by ``hidden_dim`` for local experiments."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        num_blocks: int = 3,
        top_k: int = 3,
        num_kernels: int = 6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TimesBlock(
                hidden_dim=hidden_dim,
                ff_dim=hidden_dim * 2,
                top_k=top_k,
                num_kernels=num_kernels,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch, time, {self.input_dim}], got {tuple(x.shape)}")
        encoded = self.input_projection(x)
        for block in self.blocks:
            encoded = block(encoded)
        return self.norm(encoded)[:, -1]


class SignalRawStatsFusion(nn.Module):
    """Fuse a true raw temporal encoder with an MLP statistics branch."""

    def __init__(
        self,
        raw_input_dim: int,
        stats_input_dim: int,
        encoder: str = "lstm",
        hidden_dim: int = 32,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.raw_input_dim = raw_input_dim
        self.stats_input_dim = stats_input_dim
        self.encoder_name = encoder
        self.num_classes = num_classes
        if encoder == "lstm":
            self.raw_encoder = nn.LSTM(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "gru":
            self.raw_encoder = nn.GRU(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "tcn":
            self.raw_encoder = nn.Sequential(
                CausalConv1d(raw_input_dim, hidden_dim, kernel_size=3), nn.GELU(), nn.Dropout(dropout),
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2), nn.GELU(),
            )
        elif encoder == "timesnet":
            self.raw_encoder = MiniTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif encoder == "timesnet_full":
            self.raw_encoder = FullTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown raw encoder: {encoder}")
        self.stats_encoder = nn.Sequential(
            nn.LayerNorm(stats_input_dim), nn.Linear(stats_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, raw_x: torch.Tensor, stats_x: torch.Tensor) -> torch.Tensor:
        batch_size, signal_count, _, raw_dim = raw_x.shape
        if raw_dim != self.raw_input_dim or stats_x.shape[-1] != self.stats_input_dim or stats_x.shape[-2] != 1:
            raise ValueError(f"Unexpected fusion inputs: raw={tuple(raw_x.shape)} stats={tuple(stats_x.shape)}")
        raw = raw_x.reshape(batch_size * signal_count, raw_x.shape[2], raw_dim)
        if self.encoder_name == "lstm":
            _, (hidden, _) = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "gru":
            _, hidden = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "tcn":
            raw_embedding = self.raw_encoder(raw.transpose(1, 2))[:, :, -1]
        else:
            raw_embedding = self.raw_encoder(raw)
        stats = stats_x.reshape(batch_size * signal_count, self.stats_input_dim)
        fused = torch.cat([raw_embedding, self.stats_encoder(stats)], dim=-1)
        return self.classifier(fused).reshape(batch_size, signal_count, self.num_classes)


class SignalRawStatsConditionalHeads(nn.Module):
    """Shared raw-plus-stats encoder with L1/L5-specific binary heads."""

    def __init__(
        self,
        raw_input_dim: int,
        stats_input_dim: int,
        encoder: str = "tcn",
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.stats_input_dim = stats_input_dim
        self.encoder_name = encoder
        if encoder == "lstm":
            self.raw_encoder = nn.LSTM(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "gru":
            self.raw_encoder = nn.GRU(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "tcn":
            self.raw_encoder = nn.Sequential(
                CausalConv1d(raw_input_dim, hidden_dim, kernel_size=3),
                nn.GELU(),
                nn.Dropout(dropout),
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
                nn.GELU(),
            )
        elif encoder == "timesnet":
            self.raw_encoder = MiniTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif encoder == "timesnet_full":
            self.raw_encoder = FullTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown raw encoder: {encoder}")
        self.stats_encoder = nn.Sequential(
            nn.LayerNorm(stats_input_dim),
            nn.Linear(stats_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.l1_head = nn.Linear(hidden_dim, 2)
        self.l5_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        raw_x: torch.Tensor,
        stats_x: torch.Tensor,
        is_l5: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, signal_count, _, raw_dim = raw_x.shape
        if (
            raw_dim != self.raw_input_dim
            or stats_x.shape[-1] != self.stats_input_dim
            or stats_x.shape[-2] != 1
            or tuple(is_l5.shape) != (batch_size, signal_count)
        ):
            raise ValueError(
                "Unexpected conditional fusion inputs: "
                f"raw={tuple(raw_x.shape)} stats={tuple(stats_x.shape)} is_l5={tuple(is_l5.shape)}"
            )
        raw = raw_x.reshape(batch_size * signal_count, raw_x.shape[2], raw_dim)
        if self.encoder_name == "lstm":
            _, (hidden, _) = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "gru":
            _, hidden = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "tcn":
            raw_embedding = self.raw_encoder(raw.transpose(1, 2))[:, :, -1]
        else:
            raw_embedding = self.raw_encoder(raw)
        stats = stats_x.reshape(batch_size * signal_count, self.stats_input_dim)
        shared = self.shared_fusion(torch.cat([raw_embedding, self.stats_encoder(stats)], dim=-1))
        flat_is_l5 = is_l5.reshape(-1).bool()
        logits = shared.new_empty((shared.shape[0], 2))
        logits[~flat_is_l5] = self.l1_head(shared[~flat_is_l5])
        logits[flat_is_l5] = self.l5_head(shared[flat_is_l5])
        return logits.reshape(batch_size, signal_count, 2)


class SignalRawStatsDeviceConditionalHeads(nn.Module):
    """Shared fusion encoder with routing heads for known receiver devices."""

    def __init__(
        self,
        raw_input_dim: int,
        stats_input_dim: int,
        device_ids: list[int],
        encoder: str = "tcn",
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not device_ids or len(set(device_ids)) != len(device_ids):
            raise ValueError(f"device_ids must be a non-empty unique list, got {device_ids}")
        self.raw_input_dim = raw_input_dim
        self.stats_input_dim = stats_input_dim
        self.encoder_name = encoder
        self.device_ids = tuple(sorted(device_ids))
        if encoder == "lstm":
            self.raw_encoder = nn.LSTM(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "gru":
            self.raw_encoder = nn.GRU(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "tcn":
            self.raw_encoder = nn.Sequential(
                CausalConv1d(raw_input_dim, hidden_dim, kernel_size=3),
                nn.GELU(),
                nn.Dropout(dropout),
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
                nn.GELU(),
            )
        elif encoder == "timesnet":
            self.raw_encoder = MiniTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif encoder == "timesnet_full":
            self.raw_encoder = FullTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown raw encoder: {encoder}")
        self.stats_encoder = nn.Sequential(
            nn.LayerNorm(stats_input_dim),
            nn.Linear(stats_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fallback_head = nn.Linear(hidden_dim, 2)
        self.device_heads = nn.ModuleDict({str(device_id): nn.Linear(hidden_dim, 2) for device_id in self.device_ids})

    def _shared_embedding(self, raw_x: torch.Tensor, stats_x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, signal_count, _, raw_dim = raw_x.shape
        if raw_dim != self.raw_input_dim or stats_x.shape[-1] != self.stats_input_dim or stats_x.shape[-2] != 1:
            raise ValueError(f"Unexpected device-conditional fusion inputs: raw={tuple(raw_x.shape)} stats={tuple(stats_x.shape)}")
        raw = raw_x.reshape(batch_size * signal_count, raw_x.shape[2], raw_dim)
        if self.encoder_name == "lstm":
            _, (hidden, _) = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "gru":
            _, hidden = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "tcn":
            raw_embedding = self.raw_encoder(raw.transpose(1, 2))[:, :, -1]
        else:
            raw_embedding = self.raw_encoder(raw)
        stats = stats_x.reshape(batch_size * signal_count, self.stats_input_dim)
        return self.shared_fusion(torch.cat([raw_embedding, self.stats_encoder(stats)], dim=-1)), batch_size, signal_count

    def forward(
        self,
        raw_x: torch.Tensor,
        stats_x: torch.Tensor,
        device_id: torch.Tensor,
        return_fallback: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        shared, batch_size, signal_count = self._shared_embedding(raw_x, stats_x)
        if device_id.ndim == 1 and tuple(device_id.shape) == (batch_size,):
            device_grid = device_id[:, None].expand(batch_size, signal_count)
        elif device_id.ndim == 2 and tuple(device_id.shape) == (batch_size, signal_count):
            device_grid = device_id
        else:
            raise ValueError(f"device_id must be [B] or [B,S], got {tuple(device_id.shape)}")
        fallback = self.fallback_head(shared)
        routed = fallback.clone()
        flat_device_id = device_grid.reshape(-1)
        for known_device_id in self.device_ids:
            selected = flat_device_id.eq(known_device_id)
            if selected.any():
                routed[selected] = self.device_heads[str(known_device_id)](shared[selected])
        routed = routed.reshape(batch_size, signal_count, 2)
        fallback = fallback.reshape(batch_size, signal_count, 2)
        return (routed, fallback) if return_fallback else routed

class SignalRawStatsAuxiliaryState(nn.Module):
    """Shared signal encoder with formal binary and auxiliary state heads.

    The binary head remains the deployable target-band spoof detector.  The
    three-class head is used only during training to distinguish normal,
    target-spoofed, and non-target observations inside a reviewed single-band
    attack interval.
    """

    def __init__(
        self,
        raw_input_dim: int,
        stats_input_dim: int,
        encoder: str = "tcn",
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.stats_input_dim = stats_input_dim
        self.encoder_name = encoder
        if encoder == "lstm":
            self.raw_encoder = nn.LSTM(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "gru":
            self.raw_encoder = nn.GRU(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "tcn":
            self.raw_encoder = nn.Sequential(
                CausalConv1d(raw_input_dim, hidden_dim, kernel_size=3),
                nn.GELU(),
                nn.Dropout(dropout),
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
                nn.GELU(),
            )
        elif encoder == "timesnet":
            self.raw_encoder = MiniTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif encoder == "timesnet_full":
            self.raw_encoder = FullTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown raw encoder: {encoder}")
        self.stats_encoder = nn.Sequential(
            nn.LayerNorm(stats_input_dim),
            nn.Linear(stats_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden_dim, 2)
        self.auxiliary_head = nn.Linear(hidden_dim, 3)

    def forward(self, raw_x: torch.Tensor, stats_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, signal_count, _, raw_dim = raw_x.shape
        if raw_dim != self.raw_input_dim or stats_x.shape[-1] != self.stats_input_dim or stats_x.shape[-2] != 1:
            raise ValueError(f"Unexpected auxiliary fusion inputs: raw={tuple(raw_x.shape)} stats={tuple(stats_x.shape)}")
        raw = raw_x.reshape(batch_size * signal_count, raw_x.shape[2], raw_dim)
        if self.encoder_name == "lstm":
            _, (hidden, _) = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "gru":
            _, hidden = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "tcn":
            raw_embedding = self.raw_encoder(raw.transpose(1, 2))[:, :, -1]
        else:
            raw_embedding = self.raw_encoder(raw)
        stats = stats_x.reshape(batch_size * signal_count, self.stats_input_dim)
        shared = self.shared_fusion(torch.cat([raw_embedding, self.stats_encoder(stats)], dim=-1))
        return (
            self.binary_head(shared).reshape(batch_size, signal_count, 2),
            self.auxiliary_head(shared).reshape(batch_size, signal_count, 3),
        )


class SignalRawStatsCrossBandContext(nn.Module):
    """Signal fusion classifier conditioned on same-epoch L1/L5 peer context.

    Context is calculated from active signals in the current source window:
    per-band visibility, current C/N0 and AGC means, and their changes from
    the preceding causal window history.  It is entirely receiver-observable
    at inference time and does not consume labels, TOW, or scenario metadata.
    """

    def __init__(
        self,
        raw_input_dim: int,
        stats_input_dim: int,
        cn0_index: int,
        agc_index: int,
        encoder: str = "tcn",
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.stats_input_dim = stats_input_dim
        self.cn0_index = cn0_index
        self.agc_index = agc_index
        self.encoder_name = encoder
        if encoder == "lstm":
            self.raw_encoder = nn.LSTM(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "gru":
            self.raw_encoder = nn.GRU(raw_input_dim, hidden_dim, batch_first=True)
        elif encoder == "tcn":
            self.raw_encoder = nn.Sequential(
                CausalConv1d(raw_input_dim, hidden_dim, kernel_size=3),
                nn.GELU(),
                nn.Dropout(dropout),
                CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
                nn.GELU(),
            )
        elif encoder == "timesnet":
            self.raw_encoder = MiniTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif encoder == "timesnet_full":
            self.raw_encoder = FullTimesNetEncoder(raw_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown raw encoder: {encoder}")
        self.stats_encoder = nn.Sequential(
            nn.LayerNorm(stats_input_dim),
            nn.Linear(stats_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _context(self, raw_x: torch.Tensor, mask: torch.Tensor, is_l5: torch.Tensor) -> torch.Tensor:
        """Build [L1 count/current/delta, L5 count/current/delta] per source."""
        current = raw_x[:, :, -1, :]
        history = raw_x[:, :, :-1, :].mean(dim=2) if raw_x.shape[2] > 1 else current
        active = mask.bool()
        groups: list[torch.Tensor] = []
        for band_is_l5 in (False, True):
            selected = active & is_l5.bool().eq(band_is_l5)
            weights = selected.unsqueeze(-1).to(dtype=raw_x.dtype)
            count = weights.sum(dim=1).clamp_min(1.0)
            current_selected = current[..., [self.cn0_index, self.agc_index]]
            history_selected = history[..., [self.cn0_index, self.agc_index]]
            mean_current = (current_selected * weights).sum(dim=1) / count
            mean_delta = ((current_selected - history_selected) * weights).sum(dim=1) / count
            groups.append(torch.cat([count / 32.0, mean_current, mean_delta], dim=-1))
        return torch.cat(groups, dim=-1)

    def _embedding(
        self,
        raw_x: torch.Tensor,
        stats_x: torch.Tensor,
        mask: torch.Tensor,
        is_l5: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, signal_count, _, raw_dim = raw_x.shape
        if (
            raw_dim != self.raw_input_dim
            or stats_x.shape[-1] != self.stats_input_dim
            or stats_x.shape[-2] != 1
            or tuple(mask.shape) != (batch_size, signal_count)
            or tuple(is_l5.shape) != (batch_size, signal_count)
        ):
            raise ValueError(
                "Unexpected cross-band context inputs: "
                f"raw={tuple(raw_x.shape)} stats={tuple(stats_x.shape)} "
                f"mask={tuple(mask.shape)} is_l5={tuple(is_l5.shape)}"
            )
        raw = raw_x.reshape(batch_size * signal_count, raw_x.shape[2], raw_dim)
        if self.encoder_name == "lstm":
            _, (hidden, _) = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "gru":
            _, hidden = self.raw_encoder(raw)
            raw_embedding = hidden[-1]
        elif self.encoder_name == "tcn":
            raw_embedding = self.raw_encoder(raw.transpose(1, 2))[:, :, -1]
        else:
            raw_embedding = self.raw_encoder(raw)
        stats = stats_x.reshape(batch_size * signal_count, self.stats_input_dim)
        context = self.context_encoder(self._context(raw_x, mask, is_l5))
        context = context.unsqueeze(1).expand(-1, signal_count, -1).reshape(batch_size * signal_count, -1)
        return torch.cat([raw_embedding, self.stats_encoder(stats), context], dim=-1).reshape(
            batch_size, signal_count, -1
        )

    def forward(
        self,
        raw_x: torch.Tensor,
        stats_x: torch.Tensor,
        mask: torch.Tensor,
        is_l5: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self._embedding(raw_x, stats_x, mask, is_l5))


class SignalRawStatsCrossBandContextAuxiliary(SignalRawStatsCrossBandContext):
    """E5b: cross-band context with formal and attack-associated heads."""

    def __init__(self, *args, hidden_dim: int = 32, dropout: float = 0.1, **kwargs):
        super().__init__(*args, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.binary_head = self.classifier
        del self.classifier
        self.attack_associated_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        raw_x: torch.Tensor,
        stats_x: torch.Tensor,
        mask: torch.Tensor,
        is_l5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self._embedding(raw_x, stats_x, mask, is_l5)
        return self.binary_head(embedding), self.attack_associated_head(embedding)


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
