"""FastAPI: DeepGuard baseline and/or FF++-finetuned multimodal (visual + mel + audio-only head)."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel
from safetensors.torch import load_file

import config
from inference_utils import audio_file_to_mel, image_to_tensor, video_path_to_image_mel
from models import DeepfakeDetector
from multimodal_model import MultimodalDeepfakeDetector

REPO_ID = "Harshasnade/Deepfake_Detection_System_V1"
WEIGHTS_NAME = "best_model.safetensors"
DEFAULT_MM_CKPT = config.CHECKPOINT_DIR / "multimodal_ff_best.pt"

# Default CORS: both common dev ports (see RUN_SYSTEM.md)
_DEFAULT_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:3001,http://127.0.0.1:3001"
)


def _use_multimodal_for_images() -> bool:
    """Rare: set USE_MULTIMODAL_IMAGES=1 to run image uploads through fusion head (not recommended)."""
    return os.environ.get("USE_MULTIMODAL_IMAGES", "").strip().lower() in ("1", "true", "yes")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probability_fake: float
    backend: str = "deepguard_hf"
    branch: str = "visual"


def _sigmoid_prob_fake(logit: torch.Tensor) -> float:
    return float(torch.sigmoid(logit).squeeze().item())


def _label_from_prob(prob_fake: float, threshold: float) -> tuple[str, float]:
    if prob_fake >= threshold:
        return "FAKE", prob_fake
    return "REAL", 1.0 - prob_fake


class InferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.baseline: DeepfakeDetector | None = None
        self.multimodal: MultimodalDeepfakeDetector | None = None
        self.threshold_mm = 0.5
        self.backend_image = "deepguard_hf"

    def load(self) -> None:
        weights = Path(hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_NAME))
        ckpt_path = Path(os.environ.get("MULTIMODAL_CKPT", DEFAULT_MM_CKPT))

        base_img = DeepfakeDetector(pretrained=False)
        base_img.load_state_dict(load_file(str(weights)), strict=False)
        base_img.to(self.device)
        base_img.eval()
        self.baseline = base_img
        self.backend_image = "deepguard_hf"

        self.multimodal = None
        self.threshold_mm = float(os.environ.get("MULTIMODAL_THRESHOLD", "0.5"))

        if ckpt_path.is_file():
            base_mm = DeepfakeDetector(pretrained=False)
            base_mm.load_state_dict(load_file(str(weights)), strict=False)
            base_mm.to(self.device)
            mm = MultimodalDeepfakeDetector(
                base_mm,
                visual_dim=config.VISUAL_FEATURE_DIM,
                audio_embed_dim=config.AUDIO_EMBED_DIM,
                freeze_visual=True,
            )
            try:
                data = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            except TypeError:
                data = torch.load(ckpt_path, map_location=self.device)
            mm.load_state_dict(data["state_dict"], strict=False)
            mm.to(self.device)
            mm.eval()
            self.multimodal = mm
            if os.environ.get("USE_CKPT_THRESHOLD", "").strip().lower() in ("1", "true", "yes"):
                self.threshold_mm = float(data.get("best_threshold", self.threshold_mm))

    def predict_visual_only(self, image: Image.Image) -> PredictResponse:
        t = image_to_tensor(image, self.device)
        with torch.no_grad():
            use_mm = (
                _use_multimodal_for_images()
                and self.multimodal is not None
            )
            if use_mm:
                logit = self.multimodal(t, mel=None)
                prob = _sigmoid_prob_fake(logit)
                label, conf = _label_from_prob(prob, self.threshold_mm)
                backend = "multimodal_ff"
            else:
                assert self.baseline is not None
                logit = self.baseline(t)
                prob = _sigmoid_prob_fake(logit)
                label, conf = _label_from_prob(prob, 0.5)
                backend = self.backend_image
        return PredictResponse(
            label=label,
            confidence=round(conf, 4),
            probability_fake=round(prob, 4),
            backend=backend,
            branch="visual",
        )

    def predict_video_file(self, path: Path) -> PredictResponse:
        if self.multimodal is None:
            raise RuntimeError("Video path requires finetuned multimodal checkpoint")
        img_t, mel_t = video_path_to_image_mel(path, self.device)
        with torch.no_grad():
            mask = torch.ones(1, device=self.device)
            logit = self.multimodal(img_t, mel_t, mask)
            prob = _sigmoid_prob_fake(logit)
        label, conf = _label_from_prob(prob, self.threshold_mm)
        return PredictResponse(
            label=label,
            confidence=round(conf, 4),
            probability_fake=round(prob, 4),
            backend="multimodal_ff",
            branch="multimodal_video",
        )

    def predict_audio_only_file(self, path: Path) -> PredictResponse:
        if self.multimodal is None:
            raise RuntimeError("Audio-only path requires finetuned multimodal checkpoint")
        mel_t = audio_file_to_mel(path, self.device)
        with torch.no_grad():
            logit = self.multimodal.forward_audio_only(mel_t)
            prob = _sigmoid_prob_fake(logit)
        label, conf = _label_from_prob(prob, self.threshold_mm)
        return PredictResponse(
            label=label,
            confidence=round(conf, 4),
            probability_fake=round(prob, 4),
            backend="multimodal_ff",
            branch="audio_aux",
        )


service = InferenceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(title="DeepGuard / Multimodal API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", _DEFAULT_ORIGINS).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    """Avoid 405 if you open the API root in a browser; prediction is POST-only."""
    return {
        "service": "DeepGuard API",
        "docs": "/docs",
        "health": "/health",
        "predict": {
            "path": "/predict",
            "method": "POST",
            "content_type": "multipart/form-data with field name 'file'",
            "note": "GET on /predict returns 405 Method Not Allowed — use POST from the UI or curl -F.",
        },
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(service.device),
        "image_backend": service.backend_image,
        "multimodal_loaded": service.multimodal is not None,
        "multimodal_threshold": service.threshold_mm if service.multimodal else None,
        "baseline_repo": REPO_ID,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    ct = (file.content_type or "").lower()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        if ct.startswith("image/"):
            image = Image.open(io.BytesIO(raw))
            image.load()
            return service.predict_visual_only(image)

        if ct.startswith("video/"):
            if service.multimodal is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Video needs the multimodal checkpoint. Train or place "
                        "checkpoints/multimodal_ff_best.pt (or set MULTIMODAL_CKPT)."
                    ),
                )
            suf = Path(file.filename or "clip.mp4").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
                tmp.write(raw)
                path = Path(tmp.name)
            try:
                return service.predict_video_file(path)
            finally:
                path.unlink(missing_ok=True)

        if ct.startswith("audio/"):
            if service.multimodal is None:
                raise HTTPException(
                    status_code=503,
                    detail="Audio-only needs checkpoints/multimodal_ff_best.pt (or MULTIMODAL_CKPT).",
                )
            suf = Path(file.filename or "clip.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
                tmp.write(raw)
                path = Path(tmp.name)
            try:
                return service.predict_audio_only_file(path)
            finally:
                path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Unsupported type; upload image/*, video/*, or audio/*",
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
