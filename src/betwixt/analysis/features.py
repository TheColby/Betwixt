"""Feature-stream extraction.

Each stream is a time-series that can be interpolated toward the other file's
corresponding series independently of every other stream. Streams are kept in
perceptually meaningful units (Hz, dB, ratios) rather than raw bin indices,
because perceptual.uniformity operates on them directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureStream:
    """One named feature over time."""
    name: str
    values: np.ndarray        # (frames,) or (frames, dim)
    times: np.ndarray         # (frames,) seconds
    units: str = ""
    confidence: np.ndarray | None = None


@dataclass
class FeatureSet:
    """Every extracted stream for one file, on a shared frame grid."""
    streams: dict
    sample_rate: int
    hop: int

    def __getitem__(self, name: str) -> FeatureStream:
        return self.streams[name]


def extract(x: np.ndarray, sr: int, streams: tuple, hop: int = 512,
            device: str = "cpu") -> FeatureSet:
    """Extract the requested streams from one signal.

    Uses librosa for all classical DSP features. Each extractor returns a
    FeatureStream with values on a shared frame grid.
    """
    import librosa

    # Mix to mono for feature extraction
    if x.ndim > 1:
        mono = x.mean(axis=0).astype(np.float32)
    else:
        mono = x.astype(np.float32)

    n_fft = 2048
    n_frames = 1 + len(mono) // hop
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

    extractors = {
        "rhythm": _extract_rhythm,
        "f0": _extract_f0,
        "harmonics": _extract_harmonics,
        "snr": _extract_snr,
        "centroid": _extract_centroid,
        "flatness": _extract_flatness,
        "width": _extract_width,
        "transient": _extract_transient,
        "loudness": _extract_loudness,
    }

    result = {}
    for name in streams:
        fn = extractors.get(name)
        if fn is not None:
            result[name] = fn(mono, x, sr, hop, n_fft, times, device)

    return FeatureSet(streams=result, sample_rate=sr, hop=hop)


# ---------------------------------------------------------------------------
# Per-stream extractors
# ---------------------------------------------------------------------------

def _extract_rhythm(mono, x, sr, hop, n_fft, times, device):
    import librosa
    oenv = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    # Pad or trim to match times length
    oenv = _align(oenv, len(times))
    return FeatureStream(name="rhythm", values=oenv, times=times, units="strength")


def _extract_f0(mono, x, sr, hop, n_fft, times, device):
    import librosa
    f0, voiced, _ = librosa.pyin(
        mono, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr, hop_length=hop)
    f0 = np.nan_to_num(f0, nan=0.0)
    f0 = _align(f0, len(times))
    conf = voiced.astype(np.float32) if voiced is not None else None
    if conf is not None:
        conf = _align(conf, len(times))
    return FeatureStream(name="f0", values=f0, times=times, units="Hz",
                         confidence=conf)


def _extract_harmonics(mono, x, sr, hop, n_fft, times, device):
    import librosa
    H, _ = librosa.decompose.hpss(
        librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    h_energy = np.sum(np.abs(H) ** 2, axis=0)
    h_energy = _align(h_energy, len(times))
    return FeatureStream(name="harmonics", values=h_energy, times=times,
                         units="energy")


def _extract_snr(mono, x, sr, hop, n_fft, times, device):
    import librosa
    S = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    H, P = librosa.decompose.hpss(S)
    h_pow = np.sum(np.abs(H) ** 2, axis=0)
    p_pow = np.sum(np.abs(P) ** 2, axis=0)
    total = h_pow + p_pow + 1e-12
    snr = h_pow / total
    snr = _align(snr, len(times))
    return FeatureStream(name="snr", values=snr, times=times, units="ratio")


def _extract_centroid(mono, x, sr, hop, n_fft, times, device):
    import librosa
    cent = librosa.feature.spectral_centroid(
        y=mono, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    cent = _align(cent, len(times))
    return FeatureStream(name="centroid", values=cent, times=times, units="Hz")


def _extract_flatness(mono, x, sr, hop, n_fft, times, device):
    import librosa
    flat = librosa.feature.spectral_flatness(
        y=mono, n_fft=n_fft, hop_length=hop)[0]
    flat = _align(flat, len(times))
    return FeatureStream(name="flatness", values=flat, times=times, units="ratio")


def _extract_width(mono, x, sr, hop, n_fft, times, device):
    """Stereo width as mid/side ratio. Falls back to zeros for mono."""
    if x.ndim < 2 or x.shape[0] < 2:
        return FeatureStream(name="width",
                             values=np.zeros(len(times), dtype=np.float32),
                             times=times, units="ratio")
    mid = (x[0] + x[1]) / 2.0
    side = (x[0] - x[1]) / 2.0
    # Frame-wise RMS
    frame_len = max(1, len(mid) // max(len(times), 1))
    width_vals = np.zeros(len(times), dtype=np.float32)
    for i in range(min(len(times), len(mid) // max(frame_len, 1))):
        s = i * frame_len
        e = s + frame_len
        m_rms = np.sqrt(np.mean(mid[s:e] ** 2) + 1e-12)
        s_rms = np.sqrt(np.mean(side[s:e] ** 2) + 1e-12)
        width_vals[i] = s_rms / (m_rms + s_rms)
    return FeatureStream(name="width", values=width_vals, times=times,
                         units="ratio")


def _extract_transient(mono, x, sr, hop, n_fft, times, device):
    import librosa
    oenv = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    # Transient "sharpness": derivative of onset envelope (attack slope)
    attack = np.maximum(0, np.diff(oenv, prepend=oenv[0]))
    attack = _align(attack, len(times))
    return FeatureStream(name="transient", values=attack, times=times,
                         units="slope")


def _extract_loudness(mono, x, sr, hop, n_fft, times, device):
    import librosa
    rms = librosa.feature.rms(y=mono, frame_length=n_fft, hop_length=hop)[0]
    # Convert to dB
    loud_db = 20.0 * np.log10(rms + 1e-12)
    loud_db = _align(loud_db, len(times))
    return FeatureStream(name="loudness", values=loud_db, times=times,
                         units="dB")


def _align(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim array to target length."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="edge")
