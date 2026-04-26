#!/usr/bin/env python3
"""Build frame + mel cache from FF++_Metadata.csv (run from repo with PYTHONPATH=backend)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Run as: cd Deepf2 && PYTHONPATH=backend python backend/preprocess_ff.py
_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import config
from data.ff_dataset import load_ff_metadata_rows
from data.ff_preprocess import cache_id_for_relative_path, preprocess_one_video


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ff-root", type=Path, default=config.FF_C23_ROOT)
    p.add_argument("--metadata", type=Path, default=config.FF_METADATA_CSV)
    p.add_argument("--processed-root", type=Path, default=config.PROCESSED_ROOT)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all rows with existing files")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = p.parse_args()

    rows = load_ff_metadata_rows(args.metadata, args.ff_root)
    random.Random(args.seed).shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    ok, fail = 0, 0
    for rel, vp, _y in rows:
        cid = cache_id_for_relative_path(rel)
        out = args.processed_root / cid
        if preprocess_one_video(
            vp,
            out,
            rel,
            config.IMAGE_SIZE,
            config.MEL_TIME_STEPS,
            config.N_MELS,
            args.seed,
        ):
            ok += 1
        else:
            fail += 1
            print("skip", rel)

    print(f"done ok={ok} fail={fail} processed_root={args.processed_root}")


if __name__ == "__main__":
    main()
