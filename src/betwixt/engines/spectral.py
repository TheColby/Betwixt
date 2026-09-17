"""Spectral envelope morphing -- the classical, dependency-light engine.

Approach: phase-vocoder analysis extracts the instantaneous frequency (IF)
for every STFT bin in both sources.  At each output frame the magnitudes and
IFs are interpolated; new phases are obtained by integrating the morphed IFs
from a blended initial phase.  Because the partial frequencies themselves are
being interpolated, spectral peaks *shift* from A's locations toward B's
rather than simply fading between two overlapping spectra (which would be an
amplitude crossfade).

This is still a bin-wise technique: it does not match partials across sources,
so on strongly pitched material with very different fundamentals you will still
hear a spectrally diffuse midpoint.  True partial matching lives in
SinusoidalEngine.

For material where spectral envelope (formant) shape matters more than exact
partial positions (most speech, many textural tones), cepstral-liftering
separates the slow envelope from the harmonic fine structure so each can be
morphed at its own rate.

Known limitation: blending IFs bin-wise between sources with inharmonic
partials at very different frequencies produces intermediate IFs that do not
correspond to any natural partial.  Fix: match peaks across A and B and blend
only between matched partials.  Tracked in the SinusoidalEngine roadmap.
"""

from __future__ import annotations

import numpy as np

from .base import MorphEngine, MorphRequest


# ---------------------------------------------------------------------------
# Helper: gather frames by linear interpolation (real or complex)
# ---------------------------------------------------------------------------

def _gather(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linearly interpolate S along axis-1 at fractional positions idx.

    S   : (..., n_frames)
    idx : (n_out,) float, within [0, n_frames-1]
    out : (..., n_out)
    """
    n_frames = S.shape[-1]
    lo = np.floor(idx).astype(np.intp).clip(0, n_frames - 1)
    hi = (lo + 1).clip(0, n_frames - 1)
    frac = (idx - np.floor(idx)).astype(np.float32)
    return S[..., lo] * (1.0 - frac) + S[..., hi] * frac


# ---------------------------------------------------------------------------
# Helper: tail policy
# ---------------------------------------------------------------------------

def _tail_indices(raw: np.ndarray, n_frames: int, tail: str) -> np.ndarray:
    if tail == "loop":
        return raw % n_frames
    return np.clip(raw, 0.0, n_frames - 1.0)


def _tail_alpha(alpha: np.ndarray,
                fa_raw: np.ndarray, fb_raw: np.ndarray,
                n_frames_a: int, n_frames_b: int, tail: str) -> np.ndarray:
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
        a_over = np.maximum(0.0, fa_raw - (n_frames_a - 1))
        eff = np.where(a_past, np.minimum(1.0, eff + a_over / a_len), eff)
        b_len = max(1, n_frames_b // 10)
        b_over = np.maximum(0.0, fb_raw - (n_frames_b - 1))
        eff = np.where(b_past, np.maximum(0.0, eff - b_over / b_len), eff)
    return eff


# ---------------------------------------------------------------------------
# Phase-vocoder analysis
# ---------------------------------------------------------------------------

def _pv_analysis(stft: np.ndarray, hop: int, sr: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Extract magnitude and instantaneous frequency from an STFT.

    stft  : (n_bins, n_frames) complex
    returns (mag, if_hz) each (n_bins, n_frames) float32
    """
    n_bins = stft.shape[0]
    n_fft = (n_bins - 1) * 2

    mag = np.abs(stft).astype(np.float32)
    phase = np.unwrap(np.angle(stft), axis=1).astype(np.float64)

    # Bin centre frequencies
    bin_hz = np.arange(n_bins, dtype=np.float64) * sr / n_fft  # (bins,)
    # Expected phase increment per hop at each bin
    expected_inc = 2.0 * np.pi * bin_hz * hop / sr  # (bins,)

    # Phase difference from one frame to the next
    d_phase = np.diff(phase, axis=1, prepend=phase[:, :1])  # (bins, frames)
    # Deviation from expected increment (no need to wrap – unwrapped phase)
    deviation = d_phase - expected_inc[:, np.newaxis]

    if_hz = (bin_hz[:, np.newaxis] +
             deviation * sr / (2.0 * np.pi * hop)).astype(np.float32)

    return mag, if_hz


# ---------------------------------------------------------------------------
# Cepstral spectral-envelope separation
# ---------------------------------------------------------------------------

def _cepstral_envelope(log_mag: np.ndarray, n_coeff: int = 32) -> np.ndarray:
    """Separate spectral envelope from harmonic fine structure via liftering.

    log_mag  : (n_bins, n_frames) real – log magnitude spectrum (one-sided)
    n_coeff  : number of low-quefrency cepstral coefficients to keep;
               choose so that sr / n_coeff > highest expected f0
               (n_coeff=32 @ 44100 Hz → safe up to ~1380 Hz fundamentals)
    returns  : (n_bins, n_frames) real – log spectral envelope
    """
    n_bins, n_frames = log_mag.shape
    n_fft = (n_bins - 1) * 2

    # Mirror to full symmetric spectrum
    log_full = np.concatenate([log_mag, log_mag[-2:0:-1, :]], axis=0)  # (n_fft, frames)

    # Real cepstrum
    cep = np.fft.ifft(log_full, axis=0).real  # (n_fft, frames)

    # Symmetric lifter (keep low quefrency, zero out high)
    lifter = np.zeros(n_fft)
    lifter[0] = 1.0
    lifter[1:n_coeff] = 1.0
    if n_coeff > 1:
        lifter[n_fft - n_coeff + 1:] = 1.0

    env_full = np.fft.fft(cep * lifter[:, np.newaxis], axis=0).real  # (n_fft, frames)
    return env_full[:n_bins, :]  # (n_bins, frames) log envelope


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SpectralEngine(MorphEngine):
    """STFT-based morph via phase-vocoder interpolation.

    Envelope and fine structure are separated before morphing so that
    formant peaks shift toward their counterparts in B independently of
    the harmonic fine structure.  Instantaneous frequencies are
    interpolated between sources so partial locations shift rather than
    simply fading between two superimposed spectra.
    """

    name = "spectral"
    supports = ("harmonics", "snr", "centroid", "flatness", "loudness")
    requires_torch = False

    _n_fft: int = 2048
    _hop: int = 512
    _n_coeff: int = 32   # cepstral lifter order

    def render(self, req: MorphRequest) -> np.ndarray:
        import librosa

        n_fft = self._n_fft
        hop = self._hop
        n_coeff = self._n_coeff
        eps = 1e-8

        n_out_frames = 1 + req.out_samples // hop

        # --- Alpha schedule (average supported streams) ---
        relevant = {k: v for k, v in req.alphas.items()
                    if k in self.supports and len(v) > 0}
        if not relevant:
            relevant = req.alphas

        t_out = np.linspace(0.0, 1.0, n_out_frames)

        def _stream_alpha(name: str, fallback: np.ndarray) -> np.ndarray:
            """Return per-frame alpha for a named stream, or fallback."""
            v = req.alphas.get(name)
            if v is not None and len(v) > 0:
                t_s = np.linspace(0.0, 1.0, max(len(v), 1))
                return np.interp(t_out, t_s, v)
            return fallback

        if relevant:
            resampled = []
            for v in relevant.values():
                t_src = np.linspace(0.0, 1.0, max(len(v), 1))
                resampled.append(np.interp(t_out, t_src, v))
            alpha = np.mean(resampled, axis=0).astype(np.float64)
        else:
            alpha = t_out.copy()

        # Separate stream alphas:
        #   centroid → how much the spectral envelope (formant shape) morphs
        #   harmonics → how much the harmonic fine structure morphs
        # If neither stream is active, both default to the global alpha.
        env_alpha   = _stream_alpha("centroid",  alpha)   # envelope / formants
        fine_alpha  = _stream_alpha("harmonics", alpha)   # fine structure / pitch

        t_out = np.linspace(0.0, 1.0, max(n_out_frames, 1))

        n_channels = max(req.a.shape[0], req.b.shape[0])
        out_channels = []

        for ch in range(n_channels):
            a_ch = req.a[min(ch, req.a.shape[0] - 1)].astype(np.float32)
            b_ch = req.b[min(ch, req.b.shape[0] - 1)].astype(np.float32)

            # --- STFT (center=True: zero-padded, perfect reconstruction) ---
            A = librosa.stft(a_ch, n_fft=n_fft, hop_length=hop, center=True)
            B = librosa.stft(b_ch, n_fft=n_fft, hop_length=hop, center=True)

            n_frames_a = A.shape[1]
            n_frames_b = B.shape[1]

            # --- Phase-vocoder analysis ---
            mag_A, if_A = _pv_analysis(A, hop, req.sample_rate)
            mag_B, if_B = _pv_analysis(B, hop, req.sample_rate)

            # --- Cepstral envelope separation ---
            log_mag_A = np.log(mag_A + eps)  # (bins, n_frames_a)
            log_mag_B = np.log(mag_B + eps)
            env_A = _cepstral_envelope(log_mag_A, n_coeff)
            env_B = _cepstral_envelope(log_mag_B, n_coeff)
            res_A = log_mag_A - env_A   # fine (harmonic) structure
            res_B = log_mag_B - env_B

            # --- Frame indices & tail policy ---
            fa_raw = t_out * (n_frames_a - 1)
            fb_raw = t_out * (n_frames_b - 1)
            eff_alpha = _tail_alpha(alpha, fa_raw, fb_raw,
                                    n_frames_a, n_frames_b, req.tail)
            fa = _tail_indices(fa_raw, n_frames_a, req.tail)
            fb = _tail_indices(fb_raw, n_frames_b, req.tail)

            a_bc     = eff_alpha[np.newaxis, :]              # (1, n_out) – global
            env_a_bc = env_alpha[np.newaxis, :]              # formant alpha
            fine_a_bc = fine_alpha[np.newaxis, :]            # fine-structure alpha

            # --- Gather all arrays at output positions ---
            env_A_g = _gather(env_A, fa)  # (bins, n_out)
            env_B_g = _gather(env_B, fb)
            res_A_g = _gather(res_A, fa)
            res_B_g = _gather(res_B, fb)
            if_A_g  = _gather(if_A,  fa)
            if_B_g  = _gather(if_B,  fb)

            # Initial-phase frames (integer nearest to fa[0] / fb[0])
            phase_A0 = np.angle(A[:, int(round(float(fa[0])))])
            phase_B0 = np.angle(B[:, int(round(float(fb[0])))])

            # --- Morph: envelope and fine structure use independent alphas ---
            # centroid stream → how much the spectral envelope (formants) shifts
            # harmonics stream → how much the harmonic fine structure shifts
            env_m = (1.0 - env_a_bc)  * env_A_g + env_a_bc  * env_B_g
            res_m = (1.0 - fine_a_bc) * res_A_g + fine_a_bc * res_B_g
            mag_m = np.exp(env_m + res_m).astype(np.float32)  # (bins, n_out)

            if_m  = ((1.0 - a_bc) * if_A_g  + a_bc * if_B_g).astype(np.float64)

            # --- Phase synthesis: integrate morphed IFs from blended initial ---
            a0 = float(eff_alpha[0])
            phase_0 = (1.0 - a0) * phase_A0 + a0 * phase_B0  # (bins,)

            # delta_phi[k, i] = 2π * IF_m[k, i] * hop / sr
            d_phi = 2.0 * np.pi * if_m * hop / req.sample_rate

            # Integrate: phase[:, 0] = phase_0,
            #            phase[:, i] = phase_0 + sum_{j=1}^{i} d_phi[:, j]
            cum = np.zeros_like(d_phi)
            cum[:, 1:] = np.cumsum(d_phi[:, 1:], axis=1)
            phase_m = phase_0[:, np.newaxis] + cum

            out_stft = (mag_m * np.exp(1j * phase_m)).astype(np.complex64)

            y_ch = librosa.istft(out_stft, hop_length=hop,
                                 center=True, length=req.out_samples)
            out_channels.append(y_ch.astype(np.float64))

        return np.stack(out_channels, axis=0)
