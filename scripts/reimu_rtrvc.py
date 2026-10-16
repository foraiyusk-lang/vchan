"""RVC realtime/offline wrapper that maps the input pitch onto Reimu's.

Plain RVC only transposes the input F0 by a fixed key shift, so a speaker
whose intonation swings widely (or sits far from 276 Hz) gets an output that
is expressive but not Reimu-like. The real bouyomichan female1 voice is
nearly monotone: median ~276 Hz, intonation spread ~1.3 semitones (measured
earlier in this project, see ../reimu_voice_changer). This subclass replaces
the fixed key shift with a mean-variance mapping in log-F0:

    st     = 12*log2(f0_in / user_median)      # deviation from the speaker's own median
    f0_out = 276 * 2**(st * flatten / 12)      # re-centred on Reimu's median

`flatten` in [0, 1]: 0 = fully monotone Reimu, 1 = keep the speaker's own
intonation swing (just re-centred). The user's median pitch comes from
`reimu_user_pitch.json` (written by calibrate_pitch.py) or, when absent,
from a running estimate over the voiced frames seen so far.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from infer import rtrvc

REIMU_MEDIAN_HZ = 276.0
CONFIG_PATH = Path(__file__).resolve().parent.parent / "reimu_user_pitch.json"


class ReimuPitchMap:
    def __init__(self, flatten=0.35):
        self.flatten = flatten
        self.user_median_hz = None
        self.fixed = False
        if CONFIG_PATH.exists():
            try:
                self.user_median_hz = float(json.loads(CONFIG_PATH.read_text())["median_hz"])
                self.fixed = True
            except Exception:
                pass
        self._running = []  # voiced f0 history (Hz) for the adaptive estimate

    def __call__(self, f0):
        f0 = np.asarray(f0, dtype=np.float64).copy()
        voiced = f0 > 0
        if not voiced.any():
            return f0
        if not self.fixed:
            self._running.extend(f0[voiced][::5].tolist())
            self._running = self._running[-4000:]
            self.user_median_hz = float(np.median(self._running))
        st = np.zeros_like(f0)
        st[voiced] = 12.0 * np.log2(f0[voiced] / self.user_median_hz)
        out = f0.copy()
        out[voiced] = REIMU_MEDIAN_HZ * np.power(2.0, (st[voiced] * self.flatten) / 12.0)
        return out


class ReimuRVC(rtrvc.RVC):
    """Drop-in for infer.rtrvc.RVC: F0 is remapped to Reimu's pitch range."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pitch_map = ReimuPitchMap(float(os.environ.get("REIMU_FLATTEN", "0.35")))

    def _map_and_post(self, f0_np, f0_up_key):
        f0_np = self.pitch_map(f0_np)          # re-centre on Reimu from the RAW pitch
        f0_np = f0_np * pow(2, f0_up_key / 12)  # the GUI's key slider only fine-tunes afterwards
        return self.get_f0_post(f0_np)

    def get_f0_rmvpe(self, x, f0_up_key):
        if not hasattr(self, "model_rmvpe"):
            from infer.rmvpe import RMVPE
            self.model_rmvpe = RMVPE("assets/rmvpe/rmvpe.pt", is_half=self.is_half, device=self.device)
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        uv = f0 == 0
        if np.any(~uv):
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        return self._map_and_post(f0, f0_up_key)
