"""End-to-end RVC training driver for the Reimu voice (no WebUI needed).

Stages (run any subset, in order):  preprocess -> f0 -> feature -> filelist
                                    -> train -> index
Example:
  .venv\\Scripts\\python.exe scripts\\run_training.py reimu_v1 data\\train_v1 \
          --epochs 100 --save-every 10 --batch 4
Everything runs with cwd = rvc/ and logs to logs/<exp>_<stage>.log.
Settings follow the RVC WebUI defaults for NVIDIA 16-series cards (the repo
auto-selects fp32 there): v2 model, 40k, F0 (rmvpe), pretrained f0G/f0D.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RVC = ROOT / "rvc"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
STAGES = ["preprocess", "f0", "feature", "filelist", "train", "index"]


def run(cmd, log_name):
    log_path = ROOT / "logs" / log_name
    print(">>", " ".join(cmd), "\n   log:", log_path, flush=True)
    # `-P` (py3.11+): do not prepend the script's own dir to sys.path. Without it
    # rvc/train/ shadows the `train` package with rvc/train/train.py (circular
    # import). The repo root comes from PYTHONPATH instead (infer.*, configs.*).
    cmd = [cmd[0], "-P"] + list(cmd[1:])
    # REIMU_CUDNN_BENCH=1: measured 3-4x faster steps on the GTX 1650 (2.1 -> 0.6 s/step at bs 1);
    # batch 1 is also the most efficient size within 4 GB VRAM (bs 2/4 spill and cost ~2 s/sample).
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               PYTHONPATH=str(RVC), REIMU_CUDNN_BENCH="1", REIMU_WORKERS=os.environ.get("REIMU_WORKERS", "2"))
    with open(log_path, "w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, cwd=RVC, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        print(tail)
        raise SystemExit(f"stage failed (exit {p.returncode}): {' '.join(cmd[:3])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("dataset")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sr", default="40k")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    exp = a.exp
    sr_num = {"32k": 32000, "40k": 40000, "48k": 48000}[a.sr]
    exp_dir = RVC / "logs" / exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(a.dataset)
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    stages = a.stages.split(",")

    if "preprocess" in stages:
        run([PY, "train/preprocess.py", str(dataset), str(sr_num), str(a.workers),
             str(exp_dir), "False", "3.0"], f"{exp}_preprocess.log")
    if "f0" in stages:
        run([PY, "train/dataset/extract_f0.py", "cuda", "1", "0", "0", str(exp_dir), "False"],
            f"{exp}_f0.log")
    if "feature" in stages:
        run([PY, "train/dataset/extract_hubert_feature.py", "cuda", "1", "0", "0",
             str(exp_dir), "v2", "False"], f"{exp}_feature.log")
    if "filelist" in stages:
        gt = exp_dir / "0_gt_wavs"
        feat = exp_dir / "3_feature768"
        f0d = exp_dir / "2a_f0"
        f0n = exp_dir / "2b-f0nsf"
        names = ({p.name.split(".")[0] for p in gt.iterdir()}
                 & {p.name.split(".")[0] for p in feat.iterdir()}
                 & {p.name.split(".")[0] for p in f0d.iterdir()}
                 & {p.name.split(".")[0] for p in f0n.iterdir()})
        if not names:
            raise SystemExit("no usable training segments")
        fwd = lambda p: str(p).replace("\\", "/")
        lines = [f"{fwd(gt)}/{n}.wav|{fwd(feat)}/{n}.npy|{fwd(f0d)}/{n}.wav.npy|{fwd(f0n)}/{n}.wav.npy|0"
                 for n in sorted(names)]
        mute = RVC / "logs" / "mute"
        for _ in range(2):
            lines.append(f"{fwd(mute)}/0_gt_wavs/mute{a.sr}.wav|{fwd(mute)}/3_feature768/mute.npy|"
                         f"{fwd(mute)}/2a_f0/mute.wav.npy|{fwd(mute)}/2b-f0nsf/mute.wav.npy|0")
        random.Random(0).shuffle(lines)
        (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")
        cfg_src = RVC / "configs" / ("v1" if a.sr == "40k" else "v2") / f"{a.sr}.json"
        shutil.copyfile(cfg_src, exp_dir / "config.json")
        print(f"filelist: {len(names)} segments (+mute) ; config from {cfg_src}")
    if "train" in stages:
        pre = RVC / "assets" / "pretrained_v2"
        run([PY, "train/train.py", "-e", exp, "-sr", a.sr, "-f0", "1", "-bs", str(a.batch),
             "-te", str(a.epochs), "-se", str(a.save_every),
             "-pg", f"assets/pretrained_v2/f0G{a.sr}.pth", "-pd", f"assets/pretrained_v2/f0D{a.sr}.pth",
             "-l", "1", "-c", "0", "-sw", "1", "-v", "v2"], f"{exp}_train.log")
    if "index" in stages:
        os.environ.setdefault("outside_index_root", "assets/indices")
        run([PY, "train/train_index.py", exp, "v2", "assets/indices", str(a.workers), "auto"],
            f"{exp}_index.log")


if __name__ == "__main__":
    main()
