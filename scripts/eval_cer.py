"""Rank conversion variants by how intelligible they are to Whisper.

Reference = Whisper's transcript of the UNCONVERTED speech (the same words
the speaker said); each variant is transcribed the same way and scored by
character error rate against it (lower = clearer). Relative comparison only.
Run with the system python that has faster-whisper:
  python scripts/eval_cer.py data/eval_in_25s.wav data/eval/v_*.wav
  python scripts/eval_cer.py --ref-text ref.txt data/eval/v_*.wav   # true transcript as reference
  python scripts/eval_cer.py --csv out.csv data/eval_in_25s.wav data/eval/v_*.wav
"""
import argparse, csv, os, re, sys, tempfile, wave
import numpy as np
from faster_whisper import WhisperModel

ap = argparse.ArgumentParser()
ap.add_argument("--ref-text", help="text file with the true transcript (then no reference wav is given)")
ap.add_argument("--csv", help="also write results to this CSV")
ap.add_argument("--model", default="medium")
ap.add_argument("files", nargs="+", help="[reference.wav] variant.wav ...")
args = ap.parse_args()

model = WhisperModel(args.model, device="cpu", compute_type="int8")


def transcribe(path):
    w = wave.open(path, "rb"); sr = w.getframerate(); n = w.getnframes()
    ch, sw = w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n); w.close()
    if sw != 2:
        raise SystemExit(f"{path}: need 16-bit PCM wav (got {8 * sw}-bit)")
    sig = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:  # downmix to mono
        sig = sig[: len(sig) // ch * ch].reshape(-1, ch).mean(axis=1).astype(np.int16)
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


def norm(t):
    return re.sub(r"[\s、。,.!?！？・…]", "", t)


if args.ref_text:
    ref = norm(open(args.ref_text, encoding="utf-8").read())
    variants = args.files
else:
    ref = transcribe(args.files[0])
    variants = args.files[1:]
print("REF:", ref)
rows = []
for p in variants:
    hyp = transcribe(p)
    rows.append((lev(ref, hyp) / max(1, len(ref)), os.path.basename(p), hyp))
for cer, name, hyp in sorted(rows):
    print(f"CER={cer:5.2f}  {name:22s} {hyp[:60]}")
if args.csv:
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f); wr.writerow(["cer", "file", "hypothesis"])
        wr.writerows(sorted(rows))
