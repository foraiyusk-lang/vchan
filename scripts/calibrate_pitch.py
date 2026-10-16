"""Measure your speaking pitch (median F0) once and store it for the Reimu
pitch mapping:  python scripts\calibrate_pitch.py  -> reimu_user_pitch.json
Read a sentence naturally for ~5 seconds when prompted."""
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_detect import estimate_f0_stats  # noqa: E402

SR, SECONDS = 44100, 6
print("Speak normally for", SECONDS, "seconds (e.g. read a news sentence)... recording now")
rec = sd.rec(int(SR * SECONDS), samplerate=SR, channels=1, dtype="float32")
sd.wait()
f0, st, std_st, n = estimate_f0_stats(rec[:, 0].astype(np.float64), SR, ref_hz=276.0)
if f0 is None:
    sys.exit("No voiced speech detected - move closer to the mic / speak louder.")
out = Path(__file__).resolve().parent.parent / "reimu_user_pitch.json"
out.write_text(json.dumps({"median_hz": f0, "std_semitones": std_st, "voiced_frames": n}))
print(f"median pitch {f0:.1f} Hz, intonation spread {std_st:.2f} semitones -> {out}")
