"""Paths and hyperparameters for FaceForensics++ multimodal finetuning."""

from __future__ import annotations

import os
from pathlib import Path

# Repo root (parent of backend/)
REPO_ROOT = Path(__file__).resolve().parent.parent

FF_C23_ROOT = Path(os.environ.get("FF_C23_ROOT", REPO_ROOT / "data" / "FaceForensics++_C23"))
FF_METADATA_CSV = Path(
    os.environ.get("FF_METADATA_CSV", FF_C23_ROOT / "csv" / "FF++_Metadata.csv")
)

PROCESSED_ROOT = Path(os.environ.get("PROCESSED_ROOT", REPO_ROOT / "processed_ff"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", REPO_ROOT / "checkpoints"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", REPO_ROOT / "outputs"))

IMAGE_SIZE = 224
N_MELS = 128
MEL_TIME_STEPS = 128
AUDIO_SAMPLE_RATE = 16000

# DeepGuard visual feature dim: 1280 + 128 + 64 + 768
VISUAL_FEATURE_DIM = 2240
AUDIO_EMBED_DIM = 256

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "5"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.15"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
