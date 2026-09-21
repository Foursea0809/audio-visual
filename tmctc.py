"""TM-CTC components for VoxMM AO, VO and AV experiments.

The module keeps all modalities on one time base: 25-fps mouth frames are
repeated four times to match 10-ms audio features. ``mode`` is explicit so an
AO or VO experiment cannot accidentally consume the other modality.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.body(x))


class VideoFrontEnd(nn.Module):
    """3-D convolution followed by a compact ResNet-style 2-D extractor."""
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )
        self.frame_net = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), ResidualBlock(128),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), ResidualBlock(256),
            nn.Conv2d(256, feature_dim, 3, 2, 1, bias=False), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True),
            ResidualBlock(feature_dim), nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.stem(video)
        batch, channels, frames, height, width = x.shape
        x = x.transpose(1, 2).reshape(batch * frames, channels, height, width)
        return self.frame_net(x).flatten(1).reshape(batch, frames, -1)


class TMCTCBackbone(nn.Module):
    def __init__(self, feature_dim: int = 512, audio_feature_dim: int = 80, layers: int = 6, dropout: float = .1):
        super().__init__()
        self.audio_project = nn.Linear(audio_feature_dim, feature_dim)
        self.video_project = nn.Linear(feature_dim, feature_dim)
        self.fusion_project = nn.Linear(feature_dim * 2, feature_dim)
        self.position = nn.Parameter(torch.zeros(1, 4096, feature_dim))
        layer = nn.TransformerEncoderLayer(feature_dim, 8, 2048, dropout, activation="relu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mode: str, audio: torch.Tensor | None, video: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if mode == "ao":
            if audio is None:
                raise ValueError("AO mode requires audio features.")
            x = self.audio_project(audio)
        else:
            if video is None:
                raise ValueError("VO/AV mode requires visual features.")
            visual = torch.repeat_interleave(self.video_project(video), 4, dim=1)
            if mode == "vo":
                x = visual
            elif mode == "av":
                if audio is None:
                    raise ValueError("AV mode requires audio features.")
                length = min(visual.size(1), audio.size(1))
                x = self.fusion_project(torch.cat((visual[:, :length], self.audio_project(audio[:, :length])), dim=-1))
                if padding_mask is not None:
                    padding_mask = padding_mask[:, :length]
            else:
                raise ValueError("mode must be one of: ao, vo, av")
        if x.size(1) > self.position.size(1):
            raise ValueError("Sequence exceeds positional embedding capacity (4096).")
        x = self.dropout(x + self.position[:, :x.size(1)])
        return self.encoder(x, src_key_padding_mask=padding_mask)


class TM_CTC_AVSR_Model(nn.Module):
    """Paper-oriented CTC recognizer with a manual ``ao`` / ``vo`` / ``av`` switch."""
    MODES = {"ao", "vo", "av"}

    def __init__(self, num_classes: int, mode: str = "av", audio_feature_dim: int = 80, dropout: float = .1):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("mode must be ao, vo or av")
        self.mode = mode
        self.video_frontend = VideoFrontEnd()
        self.backbone = TMCTCBackbone(audio_feature_dim=audio_feature_dim, dropout=dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, video_input: torch.Tensor | None = None, audio_input: torch.Tensor | None = None,
                mode: str | None = None, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        active_mode = mode or self.mode
        if active_mode not in self.MODES:
            raise ValueError("mode must be ao, vo or av")
        visual = self.video_frontend(video_input) if active_mode != "ao" else None
        encoded = self.backbone(active_mode, audio_input, visual, padding_mask)
        return F.log_softmax(self.classifier(encoded), dim=-1).transpose(0, 1)


if __name__ == "__main__":
    for name in ("ao", "vo", "av"):
        model = TM_CTC_AVSR_Model(num_classes=40, mode=name)
        video = None if name == "ao" else torch.randn(2, 1, 25, 112, 112)
        audio = None if name == "vo" else torch.randn(2, 100, 80)
        print(name, tuple(model(video, audio).shape))
