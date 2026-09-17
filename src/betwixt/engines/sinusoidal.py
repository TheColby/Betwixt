"""Sinusoidal-model morphing for monophonic and lightly polyphonic pitched material.

Algorithm
---------
1.  STFT both sources at n_fft=4096 (≈11.7 Hz/bin at 48 kHz — resolves
    harmonics that are 12 Hz apart).
2.  Gather output frames by proportional time mapping (same as SpectralEngine).
3.  Vectorised peak detection across all output frames via scipy maximum_filter.
4.  Per output frame:
      a. Match A↔B peaks by frequency using the Hungarian algorithm so that
         each A partial is paired with its nearest B counterpart.
      b. Morphed partial: freq = lerp(fA, fB, α), amp = lerp(aA, aB, α).
         Phase is accumulated from the previous frame so the output is
         phase-coherent (avoids clicks and phasing artefacts).
      c. Unmatched A partials fade amplitude by (1−α).
         Unmatched B partials appear with amplitude scaled by α.
      d. Non-peak bins (stochastic residual) are blended by magnitude and a
         weighted-phasor phase, exactly as SpectralEngine does.
5.  librosa.istft reconstructs the time-domain signal.

Limitation: frame-by-frame matching (no multi-frame track continuity) can
give "swimming" artefacts when partials briefly dip below the detection
threshold.  Multi-frame tracking (Kalman or nearest-neighbour linking) is the
fix; tracked as TODO.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from .base import MorphEngine, MorphRequest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gather(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linearly interpolate S[..., :] at fractional frame indices."""
    n = S.shape[-1]
    lo = np.floor(idx).astype(np.intp).clip(0, n - 1)
    hi = (lo + 1).clip(0, n - 1)
    frac = (idx - np.floor(idx)).astype(np.float32)
    return S[..., lo] * (1.0 - frac) + S[..., hi] * frac


# ---------------------------------------------------------------------------
# Vectorised peak detection
# ---------------------------------------------------------------------------

def _detect_peaks_all_frames(
    mag: np.ndarray,           # (n_bins, n_frames) float32 – linear magnitude
    min_drop_db: float = 50.0, # detect peaks ≥ (frame_peak − min_drop_db)
    min_dist: int = 2,         # minimum bins between peaks
    max_n: int = 60,           # keep at most this many peaks per frame
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (bin_indices, log_amps) per frame, sorted by amplitude desc."""
    log_mag = 20.0 * np.log10(mag + 1e-12)  # (bins, frames)

    # Local-maximum mask via max-filter along the frequency axis
    win = 2 * min_dist + 1
    local_max = maximum_filter(log_mag, size=(win, 1),
                               mode="constant", cval=-np.inf)
    is_peak = log_mag == local_max                # (bins, frames) bool

    # Per-frame relative threshold
    frame_peak = log_mag.max(axis=0)              # (frames,)
    above = log_mag >= (frame_peak - min_drop_db)

    peak_mask = is_peak & above                   # (bins, frames)

    result = []
    for i in range(mag.shape[1]):
        bins = np.where(peak_mask[:, i])[0]
        if len(bins) == 0:
            result.append((np.empty(0, int), np.empty(0)))
            continue
        order = np.argsort(log_mag[bins, i])[::-1][:max_n]
        b = bins[order]
        result.append((b, mag[b, i]))             # bins and LINEAR amps
    return result


# ---------------------------------------------------------------------------
# Per-frame peak matching
# ---------------------------------------------------------------------------

def _match_peaks(
    bins_a: np.ndarray, freqs_a: np.ndarray,  # (nA,) each
    bins_b: np.ndarray, freqs_b: np.ndarray,  # (nB,) each
    max_hz: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian matching of A peaks to B peaks by frequency proximity.

    Returns (matched_a_idx, matched_b_idx, unmatched_a_idx, unmatched_b_idx).
    """
    na, nb = len(freqs_a), len(freqs_b)
    if na == 0 or nb == 0:
        return (np.empty(0, int), np.empty(0, int),
                np.arange(na, dtype=int), np.arange(nb, dtype=int))

    cost = np.abs(freqs_a[:, None] - freqs_b[None, :])  # (nA, nB)
    cost[cost > max_hz] = 1e9

    row, col = linear_sum_assignment(cost)
    valid = cost[row, col] < max_hz
    ma, mb = row[valid], col[valid]

    matched_a_set = set(ma.tolist())
    matched_b_set = set(mb.tolist())
    ua = np.array([i for i in range(na) if i not in matched_a_set], int)
    ub = np.array([j for j in range(nb) if j not in matched_b_set], int)
    return ma, mb, ua, ub


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SinusoidalEngine(MorphEngine):
    """Phase-coherent sinusoidal morph with per-partial frequency interpolation."""

    name = "sinusoidal"
    supports = ("f0", "harmonics", "snr", "centroid", "transient", "loudness")
    requires_torch = False

    _n_fft: int = 4096
    _hop: int = 512
    _max_peaks: int = 60
    _max_hz_match: float = 100.0  # Hz — maximum frequency distance for matching
    _min_drop_db: float = 50.0    # dB below frame peak to include a partial

    def render(self, req: MorphRequest) -> np.ndarray:
        import librosa

        n_fft = self._n_fft
        hop = self._hop

        n_out_frames = 1 + req.out_samples // hop

        # --- Alpha schedule ---
        relevant = {k: v for k, v in req.alphas.items()
                    if k in self.supports and len(v) > 0}
        if not relevant:
            relevant = req.alphas

        t_out = np.linspace(0.0, 1.0, n_out_frames)
        if relevant:
            resampled = []
            for v in relevant.values():
                t_s = np.linspace(0.0, 1.0, max(len(v), 1))
                resampled.append(np.interp(t_out, t_s, v))
            alpha = np.mean(resampled, axis=0)
        else:
            alpha = t_out.copy()

        n_channels = max(req.a.shape[0], req.b.shape[0])
        out_channels = []

        for ch in range(n_channels):
            a_ch = req.a[min(ch, req.a.shape[0] - 1)].astype(np.float32)
            b_ch = req.b[min(ch, req.b.shape[0] - 1)].astype(np.float32)

            A = librosa.stft(a_ch, n_fft=n_fft, hop_length=hop, center=True)
            B = librosa.stft(b_ch, n_fft=n_fft, hop_length=hop, center=True)

            n_frames_a, n_frames_b = A.shape[1], B.shape[1]

            # Proportional frame indices + tail
            fa_raw = t_out * (n_frames_a - 1)
            fb_raw = t_out * (n_frames_b - 1)

            if req.tail == "loop":
                fa = fa_raw % n_frames_a
                fb = fb_raw % n_frames_b
            else:
                fa = np.clip(fa_raw, 0, n_frames_a - 1)
                fb = np.clip(fb_raw, 0, n_frames_b - 1)

            # Adjust alpha for drop/fade tail policy
            eff_alpha = self._tail_alpha(alpha, fa_raw, fb_raw,
                                         n_frames_a, n_frames_b, req.tail)

            # --- Gather output frames from both sources ---
            A_out = _gather(A, fa)  # (bins, n_out) complex
            B_out = _gather(B, fb)

            mag_A = np.abs(A_out).astype(np.float32)
            mag_B = np.abs(B_out).astype(np.float32)

            # --- Vectorised peak detection ---
            peaks_A = _detect_peaks_all_frames(mag_A, self._min_drop_db,
                                                min_dist=2, max_n=self._max_peaks)
            peaks_B = _detect_peaks_all_frames(mag_B, self._min_drop_db,
                                                min_dist=2, max_n=self._max_peaks)

            # --- Frame-by-frame morph ---
            out_stft = self._morph_frames(
                A_out, B_out, mag_A, mag_B,
                peaks_A, peaks_B, eff_alpha, n_fft, hop, req.sample_rate)

            y_ch = librosa.istft(out_stft, hop_length=hop,
                                 center=True, length=req.out_samples)
            out_channels.append(y_ch.astype(np.float64))

        return np.stack(out_channels, axis=0)

    # ------------------------------------------------------------------
    def _morph_frames(
        self,
        A_out: np.ndarray, B_out: np.ndarray,
        mag_A: np.ndarray, mag_B: np.ndarray,
        peaks_A: list, peaks_B: list,
        alpha: np.ndarray,
        n_fft: int, hop: int, sr: int,
    ) -> np.ndarray:
        n_bins, n_out = A_out.shape
        out = np.zeros((n_bins, n_out), dtype=np.complex64)

        bin_hz = sr / n_fft  # Hz per bin

        # Running phase accumulator: morphed_bin_int -> (phase, freq_hz)
        # We store freq_hz so we can update the phase correctly each frame.
        acc: dict[int, tuple[float, float]] = {}

        for i in range(n_out):
            alph = float(alpha[i])
            frame_a = A_out[:, i]
            frame_b = B_out[:, i]

            bins_a, amps_a = peaks_A[i]
            bins_b, amps_b = peaks_B[i]
            freqs_a = bins_a * bin_hz
            freqs_b = bins_b * bin_hz

            ma, mb, ua, ub = _match_peaks(
                bins_a, freqs_a, bins_b, freqs_b, self._max_hz_match)

            # Build sinusoidal component of output frame
            sin_frame = np.zeros(n_bins, dtype=complex)
            # Track which bins are "claimed" by sinusoidal peaks so the residual
            # can exclude them.
            sin_bins: set[int] = set()

            # Matched pairs
            for ia, ib in zip(ma, mb):
                f_a = freqs_a[ia];  a_a = amps_a[ia];  b_a = int(bins_a[ia])
                f_b = freqs_b[ib];  a_b = amps_b[ib];  b_b = int(bins_b[ib])

                f_m = (1.0 - alph) * f_a + alph * f_b
                a_m = (1.0 - alph) * a_a + alph * a_b
                b_m = int(round(f_m / bin_hz))
                b_m = max(1, min(n_bins - 2, b_m))

                phase = self._get_phase(acc, b_m, f_m,
                                        frame_a[b_a], frame_b[b_b], alph)
                sin_frame[b_m] += a_m * np.exp(1j * phase)
                acc[b_m] = (phase + 2.0 * np.pi * f_m * hop / sr, f_m)
                sin_bins.update(range(max(0, b_m - 1), min(n_bins, b_m + 2)))

            # Unmatched A partials — fade out
            for ia in ua:
                f_a = freqs_a[ia];  a_a = amps_a[ia];  b_a = int(bins_a[ia])
                phase = self._get_phase(acc, b_a, f_a, frame_a[b_a], None, alph)
                sin_frame[b_a] += a_a * (1.0 - alph) * np.exp(1j * phase)
                acc[b_a] = (phase + 2.0 * np.pi * f_a * hop / sr, f_a)
                sin_bins.update(range(max(0, b_a - 1), min(n_bins, b_a + 2)))

            # Unmatched B partials — fade in
            for ib in ub:
                f_b = freqs_b[ib];  a_b = amps_b[ib];  b_b = int(bins_b[ib])
                phase = self._get_phase(acc, b_b, f_b, None, frame_b[b_b], alph)
                sin_frame[b_b] += a_b * alph * np.exp(1j * phase)
                acc[b_b] = (phase + 2.0 * np.pi * f_b * hop / sr, f_b)
                sin_bins.update(range(max(0, b_b - 1), min(n_bins, b_b + 2)))

            # Stochastic residual: non-peak bins blended by weighted phasor
            res_mask = np.ones(n_bins, dtype=np.float32)
            if sin_bins:
                sb = np.array(list(sin_bins), dtype=int)
                sb = sb[(sb >= 0) & (sb < n_bins)]
                res_mask[sb] = 0.0

            blend = (1.0 - alph) * frame_a + alph * frame_b
            res_frame = res_mask * np.abs(blend) * np.exp(1j * np.angle(blend))

            out[:, i] = (sin_frame + res_frame).astype(np.complex64)

            # Evict stale accumulator entries (keep only those updated this frame)
            acc = {k: v for k, v in acc.items() if k in sin_bins}

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _get_phase(
        acc: dict, b: int, f_hz: float,
        spec_a,  # complex or None
        spec_b,  # complex or None
        alph: float,
    ) -> float:
        """Return accumulated phase for bin b, or initialise from sources."""
        # Search accumulator for a nearby entry (within 2 bins)
        for delta in (0, 1, -1, 2, -2):
            entry = acc.get(b + delta)
            if entry is not None:
                phase, prev_f = entry
                # Fast-forward by (b - (b+delta)) bins worth of IF
                return float(phase)

        # No history: blend initial phases from A and B
        ph_a = float(np.angle(spec_a)) if spec_a is not None else 0.0
        ph_b = float(np.angle(spec_b)) if spec_b is not None else 0.0
        return (1.0 - alph) * ph_a + alph * ph_b

    # ------------------------------------------------------------------
    @staticmethod
    def _tail_alpha(alpha, fa_raw, fb_raw, n_frames_a, n_frames_b, tail):
        if tail in ("hold", "loop"):
            return alpha
        eff = alpha.copy()
        a_past = fa_raw > (n_frames_a - 1)
        b_past = fb_raw > (n_frames_b - 1)
        if tail == "drop":
            eff = np.where(a_past, 1.0, eff)
            eff = np.where(b_past, 0.0, eff)
        elif tail == "fade":
            a_len = max(1, n_frames_a // 10)
            eff = np.where(a_past,
                           np.minimum(1.0, eff + np.maximum(0, fa_raw - (n_frames_a - 1)) / a_len),
                           eff)
            b_len = max(1, n_frames_b // 10)
            eff = np.where(b_past,
                           np.maximum(0.0, eff - np.maximum(0, fb_raw - (n_frames_b - 1)) / b_len),
                           eff)
        return eff
