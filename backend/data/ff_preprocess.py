"""Sample frames + mel spectrograms from FaceForensics++ C23 videos (cache on disk)."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import librosa
import numpy as np
from PIL import Image


def cache_id_for_relative_path(rel_path: str) -> str:
    return hashlib.sha256(rel_path.encode()).hexdigest()[:20]


def extract_wav_16k_mono(video_path: Path, out_wav: Path, duration_cap: float = 30.0) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-t",
        str(duration_cap),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_wav),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return r.returncode == 0 and out_wav.is_file() and out_wav.stat().st_size > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def wav_to_log_mel(wav_path: Path, n_mels: int = 128, sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    if y.size == 0:
        y = np.zeros(1024, dtype=np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr // 2)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = log_mel.astype(np.float32)
    return np.expand_dims(log_mel, axis=0)


def resize_mel(mel: np.ndarray, target_time: int) -> np.ndarray:
    """mel: (1, n_mels, time) -> (1, n_mels, target_time) via bilinear resize."""
    import torch
    import torch.nn.functional as F

    # (1, 1, n_mels, T)
    t = torch.from_numpy(mel).unsqueeze(0)
    t = F.interpolate(t, size=(mel.shape[1], target_time), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy().astype(np.float32)


def crop_face_bgr(frame_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = str(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    H, W = frame_bgr.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return frame_bgr[y0:y1, x0:x1]


def sample_frame_rgb(video_path: Path, frame_index: int | None = None) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release()
        return None
    idx = frame_index if frame_index is not None else max(0, n // 2)
    idx = min(idx, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    face = crop_face_bgr(frame)
    use = face if face is not None else frame
    use = cv2.cvtColor(use, cv2.COLOR_BGR2RGB)
    return use


def preprocess_one_video(
    video_path: Path,
    out_dir: Path,
    rel_key: str,
    image_size: int,
    mel_time: int,
    n_mels: int,
    seed: int,
) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = out_dir / "frame.png"
    mel_path = out_dir / "mel.npy"
    if frame_path.is_file() and mel_path.is_file():
        return True

    rng = np.random.default_rng(abs(hash(rel_key)) % (2**31) + seed)
    n_frames = 0
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    fi = int(rng.integers(0, max(1, n_frames))) if n_frames > 0 else None
    rgb = sample_frame_rgb(video_path, frame_index=fi)
    if rgb is None:
        return False

    pil = Image.fromarray(rgb).convert("RGB")
    pil = pil.resize((image_size, image_size), Image.BICUBIC)
    pil.save(frame_path)

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "clip.wav"
        if not extract_wav_16k_mono(video_path, wav):
            silent = np.zeros(16000, dtype=np.int16)
            with wave.open(str(wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(silent.tobytes())
        mel = wav_to_log_mel(wav, n_mels=n_mels)
        mel = resize_mel(mel, mel_time)
        np.save(mel_path, mel)

    return True
