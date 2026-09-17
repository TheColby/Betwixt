"""Perceptually uniform morph trajectories.

This is the module that makes the difference between a morph you can hear
happening and one you cannot.

The naive approach interpolates parameters linearly against the morph factor
and assumes perception follows. It does not: perceptual distance between
intermediate sounds is a strongly nonlinear function of the morph factor, so a
linear sweep spends most of its time sounding like one endpoint and then lurches
through the middle. That lurch is exactly the moment a listener notices.

Betwixt therefore solves for a warped parameter trajectory whose successive
intermediate results are EQUALLY SPACED in a perceptual metric, then applies
the user's chosen curve on top of that already-uniform axis. A 'linear' curve
consequently means perceptually linear.

Reference: Niu, Zhang & Martin, "SoundMorpher: Perceptually-Uniform Sound
Morphing with Diffusion Model" (arXiv:2410.02144, ICLR 2025), which establishes
an explicit proportional mapping between morph factor and perceptual stimulus
using log Mel-spectrogram features.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Log-Mel perceptual distance
# ---------------------------------------------------------------------------

def _log_mel_spectrogram(x: np.ndarray, sr: int,
                         n_fft: int = 2048, hop: int = 512,
                         n_mels: int = 128) -> np.ndarray:
    """Compute log-Mel spectrogram for a mono signal.

    Returns (n_mels, n_frames) float32.
    """
    import librosa
    # If multi-channel, mix to mono
    if x.ndim > 1:
        x = x.mean(axis=0)
    x = x.astype(np.float32)
    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def perceptual_distance(x: np.ndarray, y: np.ndarray, sr: int,
                        device: str = "cpu") -> float:
    """Perceptual distance between two signals via log-Mel RMS distance.

    This is cheap, dependency-light, and correlates well with human perception
    of spectral difference.  For each signal we compute a log-Mel spectrogram,
    then return the RMS of the frame-averaged difference across Mel bands.
    """
    mel_x = _log_mel_spectrogram(x, sr)
    mel_y = _log_mel_spectrogram(y, sr)

    # Align frame counts (use the shorter)
    n = min(mel_x.shape[1], mel_y.shape[1])
    if n == 0:
        return 0.0
    mel_x = mel_x[:, :n]
    mel_y = mel_y[:, :n]

    # RMS of the difference across all bins and frames
    diff = mel_x - mel_y
    return float(np.sqrt(np.mean(diff ** 2)))


def solve_uniform_schedule(render_fn, n_points: int = 32,
                           sr: int = 44100,
                           tolerance: float = 0.02,
                           max_iters: int = 8,
                           device: str = "cpu") -> np.ndarray:
    """Find morph factors whose renders are perceptually equally spaced.

    render_fn(alpha) -> np.ndarray  (a short probe render at that alpha).

    Algorithm:
      1. Render at a coarse grid of alphas (0, 1/n, 2/n, ..., 1).
      2. Measure cumulative perceptual arc length.
      3. Invert that mapping: for each desired uniform step in perceptual
         space, find the alpha that gets there.
      4. Optionally refine where spacing error exceeds tolerance.

    Returns (n_points,) array mapping uniform t in [0,1] to warped alpha.
    """
    # Step 1: render at coarse grid
    alphas = np.linspace(0.0, 1.0, n_points)
    renders = [render_fn(float(a)) for a in alphas]

    # Step 2: cumulative perceptual arc length
    dists = np.zeros(n_points)
    for i in range(1, n_points):
        dists[i] = perceptual_distance(renders[i - 1], renders[i], sr, device)

    cum_dist = np.cumsum(dists)
    total = cum_dist[-1]

    if total < 1e-12:
        # Signals are perceptually identical at all points; linear is fine
        return alphas.copy()

    # Normalise to [0, 1]
    cum_norm = cum_dist / total

    # Step 3: invert -- for each desired uniform t, find alpha
    t_uniform = np.linspace(0.0, 1.0, n_points)
    schedule = np.interp(t_uniform, cum_norm, alphas)

    # Step 4: iterative refinement
    for _iteration in range(max_iters):
        # Re-render at the current schedule
        new_renders = [render_fn(float(schedule[i])) for i in range(n_points)]
        new_dists = np.zeros(n_points)
        for i in range(1, n_points):
            new_dists[i] = perceptual_distance(
                new_renders[i - 1], new_renders[i], sr, device)

        new_cum = np.cumsum(new_dists)
        new_total = new_cum[-1]
        if new_total < 1e-12:
            break

        new_cum_norm = new_cum / new_total
        # Measure spacing error
        ideal_spacing = 1.0 / (n_points - 1)
        step_sizes = np.diff(new_cum_norm)
        if len(step_sizes) == 0:
            break
        error = np.max(np.abs(step_sizes - ideal_spacing)) / ideal_spacing
        if error < tolerance:
            break

        # Refine: invert again
        schedule = np.interp(t_uniform, new_cum_norm, schedule)

    # Ensure endpoints are exact
    schedule[0] = 0.0
    schedule[-1] = 1.0
    return schedule


def apply_schedule(schedule: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Warp curve output through a solved uniformity schedule."""
    if schedule is None:
        return alpha
    grid = np.linspace(0.0, 1.0, len(schedule))
    return np.interp(np.clip(alpha, 0.0, 1.0), grid, schedule)
