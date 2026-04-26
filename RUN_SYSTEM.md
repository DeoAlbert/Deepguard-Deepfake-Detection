# Run The System (Command-First)

Default ports avoid clashes with other apps: **API 8001**, **UI 3001**.

## Why you might see `405 Method Not Allowed`

- **`/predict` is POST only.** Opening it in the browser sends **GET** → FastAPI returns **405**.
- Use the **Next.js UI** (it POSTs correctly), or **`curl -F`**, or **`/docs`** to try the API.
- **`GET /`** explains this; **`GET /docs`** is the interactive Swagger UI.

## Quick Start Commands

```bash
cd /Users/deoalbert/Downloads/Deepf2

# Backend on port 8001
cd backend
source .venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8001
```

Second terminal:

```bash
cd /Users/deoalbert/Downloads/Deepf2/frontend
npm run dev -- --port 3001
```

Open:

- **UI:** [http://localhost:3001](http://localhost:3001)
- **API root (GET, no 405 on wrong method for “info”):** [http://127.0.0.1:8001/](http://127.0.0.1:8001/)
- **Health:** [http://127.0.0.1:8001/health](http://127.0.0.1:8001/health)
- **Swagger:** [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

The frontend reads **`frontend/.env.local`** → `NEXT_PUBLIC_API_URL=http://127.0.0.1:8001`. If you change the API port, update that file and restart `npm run dev`.

---

## How routing works (image vs video)

| Upload | Model |
|--------|--------|
| **Image** | Hugging Face **DeepGuard** baseline (`0.5` threshold) |
| **Video** | **Multimodal** fusion (frame + mel), if `checkpoints/multimodal_ff_best.pt` exists |
| **Audio** | Multimodal **audio-only** head, same checkpoint |

Multimodal inference uses **`MULTIMODAL_THRESHOLD`** (default **`0.5`**) so you do not get “everything fake” from an old checkpoint threshold like `0.05`. To use the threshold stored in the checkpoint instead:

```bash
USE_CKPT_THRESHOLD=1 uvicorn app:app --host 127.0.0.1 --port 8001
```

Optional: run **images** through the fusion head (usually worse):

```bash
USE_MULTIMODAL_IMAGES=1 uvicorn app:app --host 127.0.0.1 --port 8001
```

Custom checkpoint path:

```bash
MULTIMODAL_CKPT=/absolute/path/to/your.pt uvicorn app:app --host 127.0.0.1 --port 8001
```

---

## Stop servers

Press `Ctrl + C` in each terminal.

---

## One-Time Setup (if needed)

### Backend

```bash
cd /Users/deoalbert/Downloads/Deepf2/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd /Users/deoalbert/Downloads/Deepf2/frontend
npm install
```

---

## Train / Finetune (FF++ C23 multimodal)

```bash
cd /Users/deoalbert/Downloads/Deepf2
PYTHONPATH=backend backend/.venv/bin/python backend/preprocess_ff.py --max-samples 500
PYTHONPATH=backend backend/.venv/bin/python backend/train_multimodal.py --epochs 8 --batch-size 8
```

Artifacts: `checkpoints/multimodal_ff_best.pt`, `outputs/multimodal_metrics.json`, `processed_ff/`.

---

## Where Everything Is

| Path | Role |
|------|------|
| `backend/app.py` | FastAPI, dual loaders (baseline + optional multimodal), `/`, `/health`, `/predict` |
| `backend/models.py` | DeepGuard |
| `backend/multimodal_model.py` | Fusion + audio head |
| `frontend/.env.local` | `NEXT_PUBLIC_API_URL` → API base |
| `frontend/src/app/DeepfakeDetector.tsx` | Upload + fetch |
| `data/FaceForensics++_C23/` | Dataset + CSV |
| `checkpoints/` | `multimodal_ff_best.pt` |
| `processed_ff/` | Preprocessed training cache |

---

## Troubleshooting

### Free ports

```bash
lsof -ti :8001 | xargs kill -9 2>/dev/null
lsof -ti :3001 | xargs kill -9 2>/dev/null
```

### Test POST (not GET)

```bash
curl -s -F "file=@/path/to/photo.jpg" http://127.0.0.1:8001/predict
```
