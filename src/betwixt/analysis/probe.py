"""Pre-analysis: cheap descriptors computed on both files before routing.

The router needs to answer one question -- what KIND of material is this? --
without paying for full feature extraction. Everything here is classical DSP,
runs on CPU in well under a second for typical files, and needs no weights.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Probe:
    """Coarse character of one file."""
    duration: float
    sample_rate: int
    channels: int
    harmonicity: float        # 0 noise-like .. 1 strongly pitched
    onset_density: float      # onsets per second
    pulse_salience: float     # 0 no periodic pulse .. 1 strong regular beat
    spectral_flatness: float  # 0 tonal .. 1 noise-like
    stationarity: float       # 0 constantly changing .. 1 steady-state drone
    polyphony: float          # estimated simultaneous pitch count
    stereo_width: float

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def is_drone(self) -> bool:
        return self.stationarity > 0.8 and self.pulse_salience < 0.2

    @property
    def is_rhythmic(self) -> bool:
        return self.pulse_salience > 0.5

    @property
    def is_monophonic_pitched(self) -> bool:
        return self.harmonicity > 0.7 and self.polyphony < 1.5


def probe(x: np.ndarray, sr: int) -> Probe:
    """Compute routing descriptors for one signal.

    All classical DSP, no learned weights:
      harmonicity       FFT-based normalized autocorrelation peak, energy-weighted
      onset_density     positive spectral-flux onset detection with adaptive threshold
      pulse_salience    peak-to-mean ratio of the onset-envelope autocorrelation
                        over the 40–240 BPM lag range
      spectral_flatness geometric / arithmetic mean of the power spectrum per frame
      stationarity      1 - normalized mean frame-to-frame spectral change
      polyphony         salient peak count in the mean power spectrum
      stereo_width      1 - |correlation(L, R)|
    """
    from scipy.signal import stft as scipy_stft, find_peaks

    mono = x.mean(axis=0) if x.shape[0] > 1 else x[0]
    mono = mono.astype(np.float64)
    n = len(mono)
    duration = max(n / sr, 1e-6)

    hop = 512
    n_fft = 2048

    # --- STFT (no boundary padding, full interior frames only) ---
    _, _, Zxx = scipy_stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                           boundary=None, padded=False)
    mag = np.abs(Zxx)  # (n_bins, n_frames)
    n_frames = mag.shape[1]

    # --- spectral_flatness ---
    mag_eps = mag + 1e-12
    log_geo = np.mean(np.log(mag_eps), axis=0)   # (n_frames,)
    arith = np.mean(mag_eps, axis=0)
    flatness_per_frame = np.exp(log_geo) / arith  # geometric / arithmetic
    spectral_flatness = float(np.mean(flatness_per_frame))

    # --- stationarity ---
    if n_frames > 1:
        col_sum = np.sum(mag, axis=0, keepdims=True) + 1e-12
        mag_norm = mag / col_sum
        changes = np.mean(np.abs(np.diff(mag_norm, axis=1)))
        stationarity = float(np.clip(1.0 - changes * 20.0, 0.0, 1.0))
    else:
        stationarity = 1.0

    # --- onset-strength envelope via positive spectral flux ---
    diff = np.diff(mag, axis=1, prepend=mag[:, :1])
    flux = np.sum(np.maximum(diff, 0.0), axis=0)  # (n_frames,)
    max_flux = flux.max()
    flux_norm = flux / (max_flux + 1e-12)

    # onset_density: count edges that cross adaptive threshold
    thresh = float(np.mean(flux_norm) + 0.5 * np.std(flux_norm))
    above = flux_norm > thresh
    n_onsets = int(np.sum(above[1:] & ~above[:-1])) + (1 if n_frames > 0 and above[0] else 0)
    onset_density = float(n_onsets / duration)

    # --- pulse_salience: autocorrelation of onset envelope ---
    ac = np.correlate(flux_norm, flux_norm, mode='full')
    mid = len(ac) // 2
    ac = ac[mid:]
    ac = ac / (ac[0] + 1e-12)

    hop_dur = hop / sr
    lag_lo = max(1, int(round((60.0 / 240.0) / hop_dur)))   # 240 BPM
    lag_hi = min(len(ac) - 1, int(round((60.0 / 40.0) / hop_dur)))  # 40 BPM

    if lag_hi > lag_lo:
        ac_range = ac[lag_lo:lag_hi]
        peak_ac = float(np.max(ac_range))
        mean_ac = float(np.mean(np.abs(ac_range))) + 1e-6
        # peak-to-mean > 1 means some periodicity; scale so PTM=5 → salience=1
        pulse_salience = float(np.clip((peak_ac / mean_ac - 1.0) / 4.0, 0.0, 1.0))
    else:
        pulse_salience = 0.0

    # --- harmonicity: FFT-based per-frame autocorrelation, energy-weighted ---
    # Fundamental range: 80–1000 Hz
    min_lag = max(1, int(sr / 1000.0))
    max_lag = min(n_fft // 2, int(sr / 80.0))
    n_fft_ac = n_fft * 2  # zero-pad to avoid circular correlation

    harmonic_vals = []
    energies = []
    probe_hop = max(hop * 4, n_fft)   # sparse sampling for speed
    for start in range(0, n - n_fft + 1, probe_hop):
        frame = mono[start:start + n_fft]
        energy = float(np.dot(frame, frame))
        if energy < 1e-12:
            harmonic_vals.append(0.0)
            energies.append(energy)
            continue
        win = frame * np.hanning(n_fft)
        X = np.fft.rfft(win, n=n_fft_ac)
        ac_f = np.fft.irfft(np.abs(X) ** 2)[:n_fft]
        ac_f = ac_f / (ac_f[0] + 1e-12)
        if max_lag > min_lag:
            peak = float(np.max(ac_f[min_lag:max_lag]))
        else:
            peak = 0.0
        harmonic_vals.append(max(0.0, peak))
        energies.append(energy)

    if harmonic_vals:
        w = np.array(energies) + 1e-12
        harmonicity = float(np.clip(
            np.sum(np.array(harmonic_vals) * w) / np.sum(w), 0.0, 1.0))
    else:
        harmonicity = 0.0

    # --- polyphony: salient peaks in the mean power spectrum ---
    mean_spec = mag.mean(axis=1)
    mean_db = 20.0 * np.log10(mean_spec + 1e-12)
    peak_thresh_db = mean_db.max() - 30.0
    peaks, _ = find_peaks(mean_db, height=peak_thresh_db, distance=4)
    polyphony = float(min(max(len(peaks) / 4.0, 0.0), 8.0))

    # --- stereo_width ---
    if x.shape[0] >= 2:
        l_ch = x[0].astype(np.float64)
        r_ch = x[1].astype(np.float64)
        if len(l_ch) > 1 and np.std(l_ch) > 0 and np.std(r_ch) > 0:
            corr = float(np.corrcoef(l_ch, r_ch)[0, 1])
        else:
            corr = 1.0
        stereo_width = float(1.0 - abs(float(np.clip(corr, -1.0, 1.0))))
    else:
        stereo_width = 0.0

    return Probe(
        duration=duration,
        sample_rate=sr,
        channels=x.shape[0],
        harmonicity=harmonicity,
        onset_density=onset_density,
        pulse_salience=float(np.clip(pulse_salience, 0.0, 1.0)),
        spectral_flatness=float(np.clip(spectral_flatness, 0.0, 1.0)),
        stationarity=float(np.clip(stationarity, 0.0, 1.0)),
        polyphony=polyphony,
        stereo_width=stereo_width,
    )
