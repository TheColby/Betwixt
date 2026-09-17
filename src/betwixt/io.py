"""Audio input and output.

Any format in, WAV out. libsndfile covers most containers; anything it
rejects is routed through ffmpeg so that mp3, m4a, opus and video containers
all work without the user thinking about it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioLoadError(RuntimeError):
    pass


def load(path: str | Path, target_sr: int | None = None,
         mono: bool = False) -> tuple[np.ndarray, int]:
    """Read audio as (channels, samples) float64 plus sample rate."""
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"no such file: {path}")
    try:
        data, sr = sf.read(str(path), always_2d=True)
    except Exception:
        data, sr = _load_via_ffmpeg(path)

    x = data.T.astype(np.float64)
    if mono and x.shape[0] > 1:
        x = x.mean(axis=0, keepdims=True)
    if target_sr and target_sr != sr:
        x, sr = resample(x, sr, target_sr), target_sr
    return x, sr


def _load_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise AudioLoadError(
            f"could not read {path.name}; install ffmpeg for this format")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(path),
           "-c:a", "pcm_f32le", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioLoadError(
            f"ffmpeg could not decode {path.name}: "
            f"{proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}")
    data, sr = sf.read(out, always_2d=True)
    Path(out).unlink(missing_ok=True)
    return data, sr


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality resampling. Used only to reconcile input rates -- never
    as a time-scaling operation on the morph itself."""
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g, axis=-1)


def save(path: str | Path, x: np.ndarray, sr: int,
         subtype: str = "FLOAT") -> None:
    x = np.atleast_2d(x)
    sf.write(str(path), x.T, sr, subtype=subtype)


def normalize_peak(x: np.ndarray, dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return x
    return x * (10.0 ** (dbfs / 20.0)) / peak
