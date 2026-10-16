"""Rank conversion variants by how intelligible they are to Whisper.

Reference = Whisper's transcript of the UNCONVERTED speech (the same words
the speaker said); each variant is transcribed the same way and scored by
character error rate against it (lower = clearer). Relative comparison only.
Run with the system python that has faster-whisper:
  python scripts/eval_cer.py data/eval_in_25s.wav data/eval/v_*.wav
"""
import os, re, sys, tempfile, wave
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu", compute_type="int8")


def transcribe(path):
    w = wave.open(path, "rb"); sr = w.getframerate(); n = w.getnframes()
    sig = np.frombuffer(w.readframes(n), dtype=np.int16); w.close()
    text, step, d = [], int(20 * sr), tempfile.mkdtemp()
    for i, pos in enumerate(range(0, len(sig), step)):
        piece = sig[pos:pos + step]
        if len(piece) < sr * 0.3: continue
        p = os.path.join(d, f"{i}.wav")
        with wave.open(p, "wb") as o:
            o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr); o.writeframes(piece.tobytes())
        segs, _ = model.transcribe(p, language="ja", beam_size=5, vad_filter=True)
        text += [s.text for s in segs]
    return re.sub(r"[\s、。,.!?！？・…]", "", "".join(text))


def lev(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


ref = transcribe(sys.argv[1])
print("REF:", ref)
rows = []
for p in sys.argv[2:]:
    hyp = transcribe(p)
    rows.append((lev(ref, hyp) / max(1, len(ref)), os.path.basename(p), hyp))
for cer, name, hyp in sorted(rows):
    print(f"CER={cer:5.2f}  {name:22s} {hyp[:60]}")
