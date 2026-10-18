r"""Launch RVC's realtime voice-changer GUI with Reimu pitch mapping enabled.

  .venv\Scripts\python.exe scripts\reimu_realtime.py

In the GUI: pick the trained model (.pth in rvc/assets/weights) and its
.index, set the Pitch slider to 0 (pitch is re-centred on Reimu by the
mapping; the slider only fine-tunes afterwards), choose the input/output
devices, and press start. Env var REIMU_FLATTEN (default 0.35) sets how much
of your own intonation swing is kept (0 = flat Reimu, 1 = keep all of it).
Run scripts\calibrate_pitch.py once to measure your own median pitch;
otherwise it is estimated on the fly from what you say.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RVC = ROOT / "rvc"
os.chdir(RVC)
sys.path.insert(0, str(RVC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for k, v in {"weight_root": "assets/weights", "index_root": "logs",
             "outside_index_root": "assets/indices", "rmvpe_root": "assets/rmvpe",
             "REIMU_CUDNN_BENCH": "1"}.items():
    os.environ.setdefault(k, v)

from infer import rtrvc  # noqa: E402
import reimu_rtrvc       # noqa: E402

rtrvc.RVC = reimu_rtrvc.ReimuRVC  # realtime_gui.py builds RVC via rtrvc.RVC(...)
runpy.run_path(str(RVC / "realtime_gui.py"), run_name="__main__")
