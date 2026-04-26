"""PyTorch dataset over preprocessed FF++ entries (frame + mel + label)."""

from __future__ import annotations

import csv
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from data.ff_preprocess import cache_id_for_relative_path


def load_ff_metadata_rows(csv_path: Path, ff_root: Path) -> list[tuple[str, Path, int]]:
    """Returns list of (relative_path_str, absolute_video_path, label 0=real 1=fake)."""
    rows: list[tuple[str, Path, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = (row.get("File Path") or row.get("file path") or "").strip()
            if not rel:
                continue
            lab_raw = (row.get("Label") or row.get("label") or "").strip().upper()
            if lab_raw not in ("REAL", "FAKE"):
                continue
            y = 0 if lab_raw == "REAL" else 1
            vp = ff_root / rel
            if vp.is_file():
                rows.append((rel, vp, y))
    return rows


class FFMultimodalDataset(Dataset):
    def __init__(
        self,
        entries: list[tuple[str, Path, int]],
        processed_root: Path,
        augment: bool = False,
        image_size: int = 224,
    ):
        self.entries = entries
        self.processed_root = processed_root
        if augment:
            self.image_tf = A.Compose(
                [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.image_tf = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        rel, _vp, y = self.entries[i]
        cid = cache_id_for_relative_path(rel)
        base = self.processed_root / cid
        bgr = cv2.imread(str(base / "frame.png"))
        if bgr is None:
            raise FileNotFoundError(base / "frame.png")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        augmented = self.image_tf(image=img)
        image = augmented["image"]
        mel = np.load(base / "mel.npy").astype(np.float32)
        # (1, n_mels, T) — channel dim for Conv2d
        mel_t = torch.from_numpy(mel)
        return {
            "image": image,
            "mel": mel_t,
            "label": torch.tensor(y, dtype=torch.float32),
            "has_audio": torch.tensor(1.0, dtype=torch.float32),
        }
