#!/usr/bin/env python3
"""Finetune multimodal head (+ audio) on cached FF++ samples; write metrics and checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import config
from data.ff_dataset import FFMultimodalDataset, load_ff_metadata_rows
from data.ff_preprocess import cache_id_for_relative_path
from huggingface_hub import hf_hub_download
from models import DeepfakeDetector
from multimodal_model import MultimodalDeepfakeDetector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_cached_entries(processed_root: Path, all_rows: list) -> list:
    out = []
    for rel, vp, y in all_rows:
        cid = cache_id_for_relative_path(rel)
        base = processed_root / cid
        if (base / "frame.png").is_file() and (base / "mel.npy").is_file():
            out.append((rel, vp, y))
    return out


def stratified_indices(labels: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_c: dict[int, list[int]] = {0: [], 1: []}
    for i, y in enumerate(labels):
        by_c[y].append(i)
    train_idx, val_idx = [], []
    for y in (0, 1):
        idxs = by_c[y][:]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        if len(idxs) <= n_val + 1:
            n_val = max(1, len(idxs) // 5)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 46):
        pred = (probs >= t).astype(np.int32)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


@torch.no_grad()
def evaluate(
    model: MultimodalDeepfakeDetector,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict:
    model.eval()
    ys, ps_mm, ps_a = [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        mel = batch["mel"].to(device)
        mask = batch["has_audio"].to(device)
        y = batch["label"].to(device)
        logits_mm = model(img, mel, mask)
        logits_a = model.forward_audio_only(mel)
        ys.append(y.cpu().numpy())
        ps_mm.append(torch.sigmoid(logits_mm).squeeze(-1).cpu().numpy())
        ps_a.append(torch.sigmoid(logits_a).squeeze(-1).cpu().numpy())
    y_true = np.concatenate(ys)
    prob_mm = np.concatenate(ps_mm)
    prob_a = np.concatenate(ps_a)

    pred_mm = (prob_mm >= threshold).astype(np.int32)
    metrics = {
        "val_accuracy": float(accuracy_score(y_true, pred_mm)),
        "val_precision": float(precision_score(y_true, pred_mm, zero_division=0)),
        "val_recall": float(recall_score(y_true, pred_mm, zero_division=0)),
        "val_f1": float(f1_score(y_true, pred_mm, zero_division=0)),
        "val_auc_roc": float(roc_auc_score(y_true, prob_mm)) if len(np.unique(y_true)) > 1 else 0.0,
        "val_avg_precision": float(average_precision_score(y_true, prob_mm))
        if len(np.unique(y_true)) > 1
        else 0.0,
        "val_audio_only_auc": float(roc_auc_score(y_true, prob_a)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--lambda-audio", type=float, default=0.25, help="Auxiliary audio-only BCE weight")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = torch.device(args.device)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = load_ff_metadata_rows(config.FF_METADATA_CSV, config.FF_C23_ROOT)
    cached = list_cached_entries(config.PROCESSED_ROOT, all_rows)
    if len(cached) < 8:
        print(
            "Not enough cached samples. Run preprocess first, e.g.\n"
            "  PYTHONPATH=backend python backend/preprocess_ff.py --max-samples 200"
        )
        sys.exit(1)

    labels = [y for _, _, y in cached]
    train_i, val_i = stratified_indices(labels, config.VAL_RATIO, config.RANDOM_SEED)
    train_ds = FFMultimodalDataset(cached, config.PROCESSED_ROOT, augment=True)
    val_ds = FFMultimodalDataset(cached, config.PROCESSED_ROOT, augment=False)
    tr_sub = Subset(train_ds, train_i)
    drop_last = len(tr_sub) > args.batch_size
    train_loader = DataLoader(
        tr_sub,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_i),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    weights_path = hf_hub_download(
        repo_id="Harshasnade/Deepfake_Detection_System_V1",
        filename="best_model.safetensors",
    )
    base = DeepfakeDetector(pretrained=False)
    base.load_state_dict(load_file(weights_path), strict=False)
    base.to(device)

    model = MultimodalDeepfakeDetector(
        base,
        visual_dim=config.VISUAL_FEATURE_DIM,
        audio_embed_dim=config.AUDIO_EMBED_DIM,
        freeze_visual=True,
    ).to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    bce = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    best_state = None
    best_val_f1 = -1.0
    best_threshold = 0.5

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}"):
            img = batch["image"].to(device)
            mel = batch["mel"].to(device)
            mask = batch["has_audio"].to(device)
            y = batch["label"].to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            logits_mm = model(img, mel, mask)
            logits_a = model.forward_audio_only(mel)
            loss = bce(logits_mm, y) + args.lambda_audio * bce(logits_a, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # threshold from validation probabilities
        model.eval()
        ys, probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                mel = batch["mel"].to(device)
                mask = batch["has_audio"].to(device)
                y = batch["label"].numpy()
                logits = model(img, mel, mask)
                p = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                ys.append(y)
                probs.append(p)
        y_true = np.concatenate(ys)
        prob = np.concatenate(probs)
        thr, _ = best_f1_threshold(y_true, prob)
        metrics = evaluate(model, val_loader, device, thr)
        metrics["epoch"] = epoch + 1
        metrics["train_loss_mean"] = float(np.mean(losses))
        metrics["best_threshold_val"] = thr
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        if metrics["val_f1"] > best_val_f1:
            best_val_f1 = metrics["val_f1"]
            best_threshold = thr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = config.CHECKPOINT_DIR / "multimodal_ff_best.pt"
    metrics_path = config.OUTPUTS_DIR / "multimodal_metrics.json"
    report = {
        "checkpoint": str(ckpt_path),
        "best_threshold": best_threshold,
        "best_val_f1": best_val_f1,
        "history": history,
        "note": (
            "Metrics are on a small stratified val split from cached clips only; "
            "re-run with more preprocessed videos and tune threshold on a held-out set. "
            "val_audio_only_auc near 0.5 means the mel-only head is not yet discriminative at this scale."
        ),
        "config": {
            "ff_root": str(config.FF_C23_ROOT),
            "processed_root": str(config.PROCESSED_ROOT),
            "n_train": len(train_i),
            "n_val": len(val_i),
            "visual_feature_dim": config.VISUAL_FEATURE_DIM,
            "audio_embed_dim": config.AUDIO_EMBED_DIM,
        },
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    if best_state is not None:
        torch.save(
            {
                "state_dict": best_state,
                "best_threshold": best_threshold,
                "report": report,
            },
            ckpt_path,
        )
        print(f"Wrote {ckpt_path}")


if __name__ == "__main__":
    main()
