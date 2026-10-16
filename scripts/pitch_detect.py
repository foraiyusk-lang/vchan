"""Batch fundamental-frequency (F0) estimation via normalized
autocorrelation. Used for the voice calibration step (not real-time
critical, so a straightforward robust method is fine)."""

import numpy as np


def _frame_f0s(signal, sr, fmin, fmax, frame_size, hop, rms_thresh, corr_thresh):
    n = len(signal)
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    f0s = []
    pos = 0
    while pos + frame_size <= n:
        frame = signal[pos:pos + frame_size].astype(np.float64)
        pos += hop

        rms = np.sqrt(np.mean(frame * frame))
        if rms < rms_thresh:
            continue

        frame = frame - np.mean(frame)
        window = np.hanning(frame_size)
        frame = frame * window

        fft_len = 1
        while fft_len < 2 * frame_size:
            fft_len *= 2
        spec = np.fft.rfft(frame, n=fft_len)
        ac = np.fft.irfft(spec * np.conj(spec), n=fft_len)[:frame_size]

        if ac[0] <= 0:
            continue
        ac_norm = ac / ac[0]

        if max_lag >= len(ac_norm):
            continue
        segment = ac_norm[min_lag:max_lag + 1]
        if len(segment) == 0:
            continue

        peak_idx = int(np.argmax(segment))
        peak_val = segment[peak_idx]
        if peak_val < corr_thresh:
            continue

        lag = min_lag + peak_idx
        if lag <= 0:
            continue
        f0s.append(sr / lag)

    return f0s


def estimate_f0(signal, sr, fmin=70.0, fmax=400.0, frame_size=2048,
                 hop=1024, rms_thresh=0.01, corr_thresh=0.3):
    """Returns the median F0 (Hz) across voiced frames, or None if no
    frame was confidently voiced."""
    f0s = _frame_f0s(signal, sr, fmin, fmax, frame_size, hop, rms_thresh, corr_thresh)
    if not f0s:
        return None
    return float(np.median(f0s))


def estimate_f0_stats(signal, sr, ref_hz=276.0, fmin=70.0, fmax=600.0,
                       frame_size=None, hop=None, rms_thresh=0.01, corr_thresh=0.35):
    """Returns (median_f0_hz, median_semitone_rel_to_ref, std_semitone,
    n_voiced_frames), or (None, None, None, 0) if nothing confidently
    voiced was found. Uses a short frame/hop suited to picking up
    natural pitch movement within a few seconds of read speech (this is
    the same method used to measure the real female1 target statistics
    -- see voice_changer.py's module docstring -- so a user's result is
    directly comparable to it). std_semitone uses the IQR-based robust
    estimator (IQR/1.349) so a couple of octave-detection outliers on
    pitch-accent peaks don't blow up the spread.

    frame_size/hop default to None and are then derived from `sr`
    (scaled from a 512/256-samples-at-8000Hz reference, i.e. 64ms/32ms)
    rather than being fixed sample counts. This matters: a fixed
    frame_size=512 only leaves room for a max lag of 512 samples, which
    at 8000Hz covers fmin=70Hz fine (max_lag=114) but at the app's real
    default mic rate of 44100Hz needs max_lag=630 -- bigger than the
    512-sample frame itself, so every single frame was silently
    rejected and calibration ALWAYS fell back to the generic default
    voice assumption, never actually measuring the user's real pitch.
    This was a real, production-breaking bug: found via a genuine
    cross-voice conversion test at the production-realistic 44.1kHz
    rate (test_voice_conversion.py), not just the 8kHz self-reproduction
    benchmark, which never exercised this code path at production's
    actual sample rate."""
    if frame_size is None:
        frame_size = max(256, int(round(512.0 / 8000.0 * sr)))
    if hop is None:
        hop = max(128, int(round(256.0 / 8000.0 * sr)))
    f0s = _frame_f0s(signal, sr, fmin, fmax, frame_size, hop, rms_thresh, corr_thresh)
    if not f0s:
        return None, None, None, 0
    f0s = np.array(f0s)
    st = 12.0 * np.log2(f0s / ref_hz)
    median_f0 = float(np.median(f0s))
    median_st = float(np.median(st))
    iqr = float(np.percentile(st, 75) - np.percentile(st, 25))
    std_st = max(iqr / 1.349, 0.3)
    return median_f0, median_st, std_st, len(f0s)
