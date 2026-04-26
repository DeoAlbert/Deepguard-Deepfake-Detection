"""Shared tensor prep for API: image, video, and audio → model inputs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

import config
from data.ff_preprocess import (
    extract_wav_16k_mono,
    resize_mel,
    sample_frame_rgb,
    wav_to_log_mel,
)


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    tf = A.Compose(
        [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    rgb = np.array(image.convert("RGB"))
    t = tf(image=rgb)["image"].unsqueeze(0).to(device)
    return t


def video_path_to_image_mel(video_path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = sample_frame_rgb(video_path, frame_index=None)
    if rgb is None:
        raise ValueError("Could not read frame from video")
    pil = Image.fromarray(rgb).convert("RGB").resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC
    )
    img_t = image_to_tensor(pil, device)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        if not extract_wav_16k_mono(video_path, wav):
            wav.write_bytes(b"")
        if not wav.is_file() or wav.stat().st_size == 0:
            mel = np.zeros((1, config.N_MELS, config.MEL_TIME_STEPS), dtype=np.float32)
        else:
            mel = wav_to_log_mel(wav, n_mels=config.N_MELS)
            mel = resize_mel(mel, config.MEL_TIME_STEPS)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
    return img_t, mel_t


def audio_file_to_mel(audio_path: Path, device: torch.device) -> torch.Tensor:
    import librosa

    y, _ = librosa.load(str(audio_path), sr=config.AUDIO_SAMPLE_RATE, mono=True)
    if y.size == 0:
        y = np.zeros(2048, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.AUDIO_SAMPLE_RATE, n_mels=config.N_MELS, fmax=config.AUDIO_SAMPLE_RATE // 2
    )
    log_mel = np.expand_dims(librosa.power_to_db(mel, ref=np.max).astype(np.float32), axis=0)
    log_mel = resize_mel(log_mel, config.MEL_TIME_STEPS)
    return torch.from_numpy(log_mel).unsqueeze(0).to(device)
