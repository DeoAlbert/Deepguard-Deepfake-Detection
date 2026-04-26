"""Visual (DeepGuard) + mel-spectrogram fusion for binary deepfake detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from models import DeepfakeDetector


class MelEncoder(nn.Module):
    """Lightweight CNN on log-mel maps (B, 1, n_mels, time)."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.net(mel)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MultimodalDeepfakeDetector(nn.Module):
    """
    Frozen DeepGuard backbone for visual features; trainable mel encoder + fusion head.
    Image-only inference uses a learned null-audio embedding.
    """

    def __init__(
        self,
        visual_backbone: DeepfakeDetector,
        visual_dim: int,
        audio_embed_dim: int,
        freeze_visual: bool = True,
    ):
        super().__init__()
        self.visual = visual_backbone
        self._freeze_visual = freeze_visual
        if freeze_visual:
            for p in self.visual.parameters():
                p.requires_grad = False

        self.audio_enc = MelEncoder(in_channels=1, embed_dim=audio_embed_dim)
        self.null_audio = nn.Parameter(torch.zeros(1, audio_embed_dim))
        fused = visual_dim + audio_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )
        # Auxiliary audio-only path (for clips without reliable video)
        self.audio_only_head = nn.Sequential(
            nn.Linear(audio_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_visual:
            self.visual.eval()
        return self

    def forward(
        self,
        image: torch.Tensor,
        mel: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = image.size(0)
        if self._freeze_visual:
            with torch.no_grad():
                v = self.visual.forward_visual_features(image)
        else:
            v = self.visual.forward_visual_features(image)

        if mel is None:
            a = self.null_audio.expand(b, -1)
        else:
            a = self.audio_enc(mel)
            if audio_mask is not None:
                m = audio_mask.view(b, 1).to(dtype=a.dtype, device=a.device)
                a = a * m + self.null_audio.expand(b, -1) * (1.0 - m)

        return self.head(torch.cat([v, a], dim=1))

    def forward_audio_only(self, mel: torch.Tensor) -> torch.Tensor:
        return self.audio_only_head(self.audio_enc(mel))
