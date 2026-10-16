"""Pick a training subset from the bulk-generated Reimu corpus.

The bulk set is ~11 hours (data/wav_8k). RVC fine-tuning converges well on
roughly 1-1.5 hours of clean target speech, and every extra hour costs a lot
of wall-clock on a 4GB GTX 1650, so by default a seeded random subset of
short sentences + a few long paragraphs is copied to data/train_<name>/.
Usage: python scripts/select_dataset.py v1 --short 1100 --para 100
"""

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def dur(p: Path) -> float:
    return (p.stat().st_size - 44) / 16000.0  # 8kHz mono 16-bit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--short", type=int, default=1100)
    ap.add_argument("--para", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    src = ROOT / "data" / "wav_8k"
    dst = ROOT / "data" / f"train_{a.name}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    rng = random.Random(a.seed)
    shorts = sorted(src.glob("s*.wav"))
    paras = sorted(src.glob("p*.wav"))
    pick = rng.sample(shorts, min(a.short, len(shorts))) + rng.sample(paras, min(a.para, len(paras)))
    total = 0.0
    for p in pick:
        shutil.copy2(p, dst / p.name)
        total += dur(p)
    print(f"{len(pick)} files, {total/60:.1f} min -> {dst}")


if __name__ == "__main__":
    main()
