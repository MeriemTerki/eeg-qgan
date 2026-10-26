"""Downstream seizure detector.

This is the model that decides the clinical-utility question, so its training protocol
matters as much as its architecture. Audit item F8 lives in the trainer rather than
here: the validation split used for early stopping must be carved out before synthetic
windows are added, otherwise synthetic samples land in the validation set and the
stopping criterion is partly measuring how well the model fits the generator.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTMDetector(nn.Module):
    """Convolutional front end followed by a bidirectional LSTM.

    Convolutions capture the waveform morphology within a window; the recurrent layer
    aggregates across time. Channels are treated as input features so cross-channel
    structure is available from the first layer.
    """

    def __init__(
        self,
        n_channels: int,
        sample_length: int = 256,
        conv_widths: tuple[int, ...] = (32, 64),
        lstm_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = int(n_channels)
        for w in conv_widths:
            layers.extend(
                [
                    nn.Conv1d(in_ch, int(w), 5, stride=2, padding=2),
                    nn.BatchNorm1d(int(w)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = int(w)
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=int(lstm_hidden),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * int(lstm_hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)  # (B, C', T')
        h = h.transpose(1, 2)  # (B, T', C')
        out, _ = self.lstm(h)
        # Mean over time rather than the last state, so a window is not summarised by
        # its final instant.
        return self.head(out.mean(dim=1)).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _SqueezeExcite(nn.Module):
    """Channel gate, so a few informative feature maps can dominate the response."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, int(channels) // int(reduction))
        self.fc = nn.Sequential(
            nn.Linear(int(channels), hidden),
            nn.GELU(),
            nn.Linear(hidden, int(channels)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x.mean(dim=-1)).unsqueeze(-1)


class _ResidualBlock(nn.Module):
    """Stride-1 residual block with batch normalisation and a channel gate."""

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        pad = int(kernel_size) // 2
        self.body = nn.Sequential(
            nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad),
            nn.BatchNorm1d(int(channels)),
            nn.GELU(),
            nn.Dropout1d(float(dropout)),
            nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad),
            nn.BatchNorm1d(int(channels)),
        )
        self.gate = _SqueezeExcite(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.gate(self.body(x)))


class _AttentionPool(nn.Module):
    """Learned weighted average over time.

    A seizure occupies part of a window rather than all of it, so a plain mean over the
    recurrent output dilutes the discriminative interval with surrounding background.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(int(dim), 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(h), dim=1)
        return (h * weights).sum(dim=1)


class SpectralFeatures(nn.Module):
    """Relative band power per channel, computed from the window itself.

    Seizure activity is defined clinically by rhythmic activity in specific bands, and a
    convolutional stack has to spend capacity rediscovering that from raw samples. These
    features are a fixed, differentiable function of the input with no learned parameters
    in the transform itself, and they are applied identically to real and synthetic
    windows, so they add no information that the raw window did not already carry.
    """

    BANDS: tuple[tuple[float, float], ...] = (
        (0.5, 4.0),    # delta
        (4.0, 8.0),    # theta
        (8.0, 13.0),   # alpha
        (13.0, 30.0),  # beta
        (30.0, 70.0),  # low gamma
    )

    def __init__(self, n_channels: int, sample_length: int, target_fs: float = 200.0) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        freqs = torch.fft.rfftfreq(int(sample_length), d=1.0 / float(target_fs))
        masks = torch.stack(
            [((freqs >= lo) & (freqs < hi)).to(torch.float32) for lo, hi in self.BANDS]
        )
        # Buffers move with the module, so the features follow the model onto the GPU.
        self.register_buffer("masks", masks, persistent=False)
        self.n_features = self.n_channels * (len(self.BANDS) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        power = torch.fft.rfft(x, dim=-1).abs().pow(2)          # (B, C, F)
        band = torch.einsum("bcf,kf->bck", power, self.masks)   # (B, C, K)
        total = band.sum(dim=-1, keepdim=True)
        relative = band / (total + 1e-8)
        # Absolute scale carries amplitude information that the ratios discard.
        log_total = torch.log1p(total)
        return torch.cat([relative, log_total], dim=-1).flatten(start_dim=1)


class CNNBiLSTMDetectorV2(nn.Module):
    """Residual convolutional encoder, recurrent aggregation and attention pooling.

    Motivated by the ae_v5c probe, where the shallow two-layer stack reached an AUROC of
    0.67 on the harder holdout and 0.74 on the easier one. The changes address the three
    things that stack could not do: represent morphology at more than one scale, weight
    the part of the window where the event actually sits, and see band structure directly.
    """

    def __init__(
        self,
        n_channels: int,
        sample_length: int = 256,
        conv_widths: tuple[int, ...] = (64, 128, 256),
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        conv_dropout: float = 0.1,
        blocks_per_stage: int = 1,
        use_spectral: bool = True,
        target_fs: float = 200.0,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        widths = tuple(int(w) for w in conv_widths)
        first = widths[0]
        self.stem = nn.Sequential(
            nn.Conv1d(int(n_channels), first, 7, padding=3),
            nn.BatchNorm1d(first),
            nn.GELU(),
        )

        stages: list[nn.Module] = []
        in_ch = first
        for w in widths:
            layers: list[nn.Module] = [
                nn.Conv1d(in_ch, int(w), 5, stride=2, padding=2),
                nn.BatchNorm1d(int(w)),
                nn.GELU(),
            ]
            layers.extend(
                _ResidualBlock(int(w), dropout=float(conv_dropout))
                for _ in range(max(0, int(blocks_per_stage)))
            )
            stages.append(nn.Sequential(*layers))
            in_ch = int(w)
        self.stages = nn.Sequential(*stages)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=int(lstm_hidden),
            num_layers=max(1, int(lstm_layers)),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )
        rec_dim = 2 * int(lstm_hidden)
        self.attend = _AttentionPool(rec_dim)

        # Attention, mean and max summaries answer different questions: where the model
        # thinks the event is, what the window looks like overall, and how extreme its
        # most abnormal instant was.
        pooled_dim = 3 * rec_dim
        self.spectral = (
            SpectralFeatures(n_channels, sample_length, target_fs) if use_spectral else None
        )
        if self.spectral is not None:
            self.spectral_proj = nn.Sequential(
                nn.LayerNorm(self.spectral.n_features),
                nn.Linear(self.spectral.n_features, int(head_hidden)),
                nn.GELU(),
            )
            pooled_dim += int(head_hidden)

        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(pooled_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.stem(x)).transpose(1, 2)
        out, _ = self.lstm(h)
        pooled = torch.cat([self.attend(out), out.mean(dim=1), out.max(dim=1).values], dim=-1)
        if self.spectral is not None:
            pooled = torch.cat([pooled, self.spectral_proj(self.spectral(x))], dim=-1)
        return self.head(pooled).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LatentMLPDetector(nn.Module):
    """Classifier over latent codes, for the decoder-free evaluation path."""

    def __init__(self, latent_dim: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(latent_dim)
        for h in hidden:
            layers.extend([nn.Linear(prev, int(h)), nn.ReLU(), nn.Dropout(dropout)])
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class STFTCNNDetector(nn.Module):
    """Log-magnitude STFT + 2-D CNN for longer clinical clips (≈8–12 s).

    Input remains a waveform ``(B, C, T)``. The STFT is computed inside the model so
    real and synthetic clips share the same transform. This is the literature-style
    clinical protocol companion to the 1.28 s waveform rematch detector — not a
    replacement for the generative study.
    """

    def __init__(
        self,
        n_channels: int,
        sample_length: int = 2048,
        n_fft: int = 128,
        hop_length: int = 64,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.3,
        head_hidden: int = 128,
        target_fs: float = 200.0,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_length = int(sample_length)
        self.target_fs = float(target_fs)
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
            persistent=False,
        )

        layers: list[nn.Module] = []
        in_ch = int(n_channels)
        for w in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, int(w), kernel_size=3, padding=1),
                    nn.BatchNorm2d(int(w)),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(float(dropout) * 0.5),
                ]
            )
            in_ch = int(w)
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(in_ch),
            nn.Dropout(float(dropout)),
            nn.Linear(in_ch, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), 1),
        )

    def _log_stft(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        flat = x.reshape(b * c, t)
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=x.dtype, device=x.device),
            return_complex=True,
            center=True,
        )
        mag = spec.abs()
        return torch.log1p(mag).reshape(b, c, mag.shape[-2], mag.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(self._log_stft(x))).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class STFTBiLSTMDetector(nn.Module):
    """Spectrogram front end that pools frequency and keeps time, then aggregates.

    ``STFTCNNDetector`` ends in ``AdaptiveAvgPool2d((1, 1))``, which averages the whole
    time-frequency plane into one vector. On a 1.28 s window that costs little. On a
    10 s clip it discards the evolution — onset, rhythmic build-up, spread — which is
    the entire reason for looking at a longer clip. Here the convolutional stages pool
    only along frequency, so the recurrent layer sees one feature vector per STFT frame
    and attention pooling can concentrate on the frames where the event actually sits.
    """

    def __init__(
        self,
        n_channels: int,
        sample_length: int = 2048,
        n_fft: int = 128,
        hop_length: int = 64,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        proj_dim: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        conv_dropout: float = 0.1,
        head_hidden: int = 256,
        use_spectral: bool = True,
        target_fs: float = 200.0,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_length = int(sample_length)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

        # Normalising over the batch rather than within a clip keeps absolute band
        # power comparable across clips, which is what separates a seizure from
        # background once recording-scope amplitude survives preprocessing.
        self.input_norm = nn.BatchNorm2d(int(n_channels))

        n_freq = self.n_fft // 2 + 1
        stages: list[nn.Module] = []
        in_ch = int(n_channels)
        for w in conv_channels:
            stages.extend(
                [
                    nn.Conv2d(in_ch, int(w), kernel_size=3, padding=1),
                    nn.BatchNorm2d(int(w)),
                    nn.GELU(),
                    nn.Conv2d(int(w), int(w), kernel_size=3, padding=1),
                    nn.BatchNorm2d(int(w)),
                    nn.GELU(),
                    nn.MaxPool2d((2, 1)),
                    nn.Dropout2d(float(conv_dropout)),
                ]
            )
            in_ch = int(w)
            n_freq = max(1, n_freq // 2)
        self.conv = nn.Sequential(*stages)

        self.project = nn.Sequential(
            nn.Conv1d(in_ch * n_freq, int(proj_dim), kernel_size=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=int(proj_dim),
            hidden_size=int(lstm_hidden),
            num_layers=max(1, int(lstm_layers)),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )
        rec_dim = 2 * int(lstm_hidden)
        self.attend = _AttentionPool(rec_dim)
        pooled_dim = 3 * rec_dim

        self.spectral = (
            SpectralFeatures(n_channels, self.sample_length, target_fs)
            if use_spectral
            else None
        )
        if self.spectral is not None:
            self.spectral_proj = nn.Sequential(
                nn.LayerNorm(self.spectral.n_features),
                nn.Linear(self.spectral.n_features, int(head_hidden)),
                nn.GELU(),
            )
            pooled_dim += int(head_hidden)

        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(pooled_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), 1),
        )

    def _log_stft(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        spec = torch.stft(
            x.reshape(b * c, t),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=x.dtype, device=x.device),
            return_complex=True,
            center=True,
        )
        mag = spec.abs()
        return torch.log1p(mag).reshape(b, c, mag.shape[-2], mag.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(self.input_norm(self._log_stft(x)))  # (B, W, F', Tf)
        b, w, f, tf = h.shape
        h = self.project(h.reshape(b, w * f, tf)).transpose(1, 2)  # (B, Tf, proj)
        out, _ = self.lstm(h)
        pooled = torch.cat(
            [self.attend(out), out.mean(dim=1), out.max(dim=1).values], dim=-1
        )
        if self.spectral is not None:
            pooled = torch.cat([pooled, self.spectral_proj(self.spectral(x))], dim=-1)
        return self.head(pooled).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FocalLoss(nn.Module):
    """Focal loss for the class imbalance that motivates the whole project.

    Down-weights easy negatives so the gradient is dominated by the rare positives.
    ``pos_weight`` is kept separate so the imbalance can be addressed by reweighting,
    by focusing, or by both, and the choice is recorded.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = (
            torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            if self.pos_weight is not None
            else None
        )
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weight, reduction="none"
        )
        p_t = torch.exp(-bce.clamp(max=20.0))
        return ((1.0 - p_t) ** self.gamma * bce).mean()


def build_detector(
    cfg: dict,
    n_channels: int,
    sample_length: int | None = None,
) -> nn.Module:
    det_cfg = cfg.get("detector", {}) or {}
    kind = str(det_cfg.get("kind", "cnn_bilstm_v2"))
    length = int(
        sample_length
        if sample_length is not None
        else det_cfg.get("sample_length", cfg.get("sample_length", 256))
    )
    if kind == "cnn_bilstm_v2":
        return CNNBiLSTMDetectorV2(
            n_channels=int(n_channels),
            sample_length=length,
            conv_widths=tuple(det_cfg.get("conv_widths", (64, 128, 256))),
            lstm_hidden=int(det_cfg.get("lstm_hidden", 128)),
            lstm_layers=int(det_cfg.get("lstm_layers", 1)),
            dropout=float(det_cfg.get("dropout", 0.3)),
            conv_dropout=float(det_cfg.get("conv_dropout", 0.1)),
            blocks_per_stage=int(det_cfg.get("blocks_per_stage", 1)),
            use_spectral=bool(det_cfg.get("use_spectral", True)),
            target_fs=float(cfg.get("target_fs", 200)),
            head_hidden=int(det_cfg.get("head_hidden", 128)),
        )
    if kind == "cnn_bilstm":
        return CNNBiLSTMDetector(
            n_channels=int(n_channels),
            sample_length=length,
            conv_widths=tuple(det_cfg.get("conv_widths", (32, 64))),
            lstm_hidden=int(det_cfg.get("lstm_hidden", 64)),
            dropout=float(det_cfg.get("dropout", 0.3)),
        )
    if kind == "stft_cnn":
        return STFTCNNDetector(
            n_channels=int(n_channels),
            sample_length=length,
            n_fft=int(det_cfg.get("n_fft", (cfg.get("clinical") or {}).get("n_fft", 128))),
            hop_length=int(
                det_cfg.get("hop_length", (cfg.get("clinical") or {}).get("hop_length", 64))
            ),
            conv_channels=tuple(det_cfg.get("conv_channels", (32, 64, 128))),
            dropout=float(det_cfg.get("dropout", 0.3)),
            head_hidden=int(det_cfg.get("head_hidden", 128)),
            target_fs=float(cfg.get("target_fs", 200)),
        )
    if kind == "stft_bilstm":
        clinical = cfg.get("clinical") or {}
        return STFTBiLSTMDetector(
            n_channels=int(n_channels),
            sample_length=length,
            n_fft=int(det_cfg.get("n_fft", clinical.get("n_fft", 128))),
            hop_length=int(det_cfg.get("hop_length", clinical.get("hop_length", 64))),
            conv_channels=tuple(det_cfg.get("conv_channels", (32, 64, 128))),
            proj_dim=int(det_cfg.get("proj_dim", 256)),
            lstm_hidden=int(det_cfg.get("lstm_hidden", 128)),
            lstm_layers=int(det_cfg.get("lstm_layers", 2)),
            dropout=float(det_cfg.get("dropout", 0.3)),
            conv_dropout=float(det_cfg.get("conv_dropout", 0.1)),
            head_hidden=int(det_cfg.get("head_hidden", 256)),
            use_spectral=bool(det_cfg.get("use_spectral", True)),
            target_fs=float(cfg.get("target_fs", 200)),
        )
    if kind == "latent_mlp":
        return LatentMLPDetector(
            latent_dim=int(cfg["latent_dim"]),
            hidden=tuple(det_cfg.get("hidden", (64, 32))),
            dropout=float(det_cfg.get("dropout", 0.2)),
        )
    raise ValueError(
        f"Unknown detector kind '{kind}'; expected cnn_bilstm_v2, cnn_bilstm, "
        "stft_cnn, stft_bilstm or latent_mlp"
    )
