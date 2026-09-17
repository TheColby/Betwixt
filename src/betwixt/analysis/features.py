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

    TODO: implement per-stream extractors:
      rhythm      onset-strength envelope, onset times, tempo grid
      f0          pretrained tracker (torchcrepe) with confidence; for
                  polyphonic material a dominant-pitch salience map instead
      harmonics   amplitudes at k*f0, or HPSS harmonic-component envelope
                  when polyphonic
      snr         harmonic vs stochastic energy ratio per band (HPSS)
      centroid    energy-weighted mean frequency, Hz then Bark
      flatness    per-band spectral flatness
      width       per-band L/R correlation and side/mid ratio
      transient   attack slope distribution, independent of onset placement
      loudness    ITU-R BS.1770 short-term loudness contour
    """
    raise NotImplementedError("analysis.features.extract")
