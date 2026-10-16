"""Offline voice conversion (speaker voice -> trained Reimu model) for
evaluating checkpoints on the recordings made with the earlier project.

  .venv\\Scripts\\python.exe scripts\\convert_offline.py <model.pth> <in.wav> <out.wav>
        [--index added_xxx.index] [--index-rate 0.5] [--flatten 0.35] [--key 0]

Pitch handling = ReimuPitchMap (see reimu_rtrvc.py): the speaker's F0 is
re-centred on Reimu's 276 Hz and its swing scaled by --flatten, replacing
RVC's plain key shift. The speaker's own median is measured from the input.
"""

import argparse
import os
import sys
from pathlib import Path

START_CWD = Path.cwd()  # relative CLI paths mean "relative to where the user ran this"
ROOT = Path(__file__).resolve().parent.parent
RVC = ROOT / "rvc"
sys.path.insert(0, str(RVC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(RVC)
os.environ.setdefault("weight_root", str(RVC / "assets" / "weights"))
os.environ.setdefault("index_root", str(RVC / "logs"))
os.environ.setdefault("outside_index_root", str(RVC / "assets" / "indices"))
os.environ.setdefault("rmvpe_root", str(RVC / "assets" / "rmvpe"))

import numpy as np  # noqa: E402

from infer.vc import pipeline as vc_pipeline  # noqa: E402

REIMU_MEDIAN_HZ = 276.0


def install_pitch_map(flatten):
    orig = vc_pipeline.Pipeline.get_f0
    state = {"median": None}

    def patched(self, x, p_len, f0_up_key, f0_method):
        _coarse, f0 = orig(self, x, p_len, 0, f0_method)  # raw Hz, no key shift
        f0 = np.asarray(f0, dtype=np.float64)
        voiced = f0 > 0
        if voiced.any():
            if state["median"] is None:
                state["median"] = float(np.median(f0[voiced]))
            st = 12.0 * np.log2(f0[voiced] / state["median"])
            f0[voiced] = REIMU_MEDIAN_HZ * np.power(2.0, st * flatten / 12.0)
        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_min, f0_max = 50, 1100
        mel_min = 1127 * np.log(1 + f0_min / 700)
        mel_max = 1127 * np.log(1 + f0_max / 700)
        mel = 1127 * np.log(1 + f0 / 700)
        mel[mel > 0] = (mel[mel > 0] - mel_min) * 254 / (mel_max - mel_min) + 1
        mel[mel <= 1] = 1
        mel[mel > 255] = 255
        return np.rint(mel).astype(np.int32), f0bak

    vc_pipeline.Pipeline.get_f0 = patched
    return state


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else (START_CWD / p).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--index", default="")
    ap.add_argument("--index-rate", type=float, default=0.5)
    ap.add_argument("--flatten", type=float, default=0.35)
    ap.add_argument("--key", type=int, default=0)
    ap.add_argument("--protect", type=float, default=0.33)
    a = ap.parse_args()

    install_pitch_map(a.flatten)

    # Reuse the stock CLI entry so model loading / chunking / writing stay as upstream
    from infer import cli
    argv = ["--model", str(_abs(a.model)), "--input", str(_abs(a.input)),
            "--output", str(_abs(a.output)), "--pitch", str(a.key),
            "--index-rate", str(a.index_rate), "--protect", str(a.protect), "--overwrite"]
    if a.index:
        argv += ["--index", str(_abs(a.index))]
    cli.main(argv)


if __name__ == "__main__":
    main()
