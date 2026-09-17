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


def perceptual_distance(x: np.ndarray, y: np.ndarray, sr: int,
                        device: str = "cpu") -> float:
    """Perceptual distance between two signals.

    TODO: implement. Planned: log-Mel spectral distance as the cheap default,
    with CDPAM as an optional higher-fidelity metric when torch is available.
    """
    raise NotImplementedError("perceptual.uniformity.perceptual_distance")


def solve_uniform_schedule(render_fn, n_points: int = 32,
                           tolerance: float = 0.02,
                           max_iters: int = 8) -> np.ndarray:
    """Find morph factors whose renders are perceptually equally spaced.

    render_fn(alpha) -> a short probe render at that morph position.

    Returns an array mapping normalized time to the morph factor that should
    be used there, so downstream code can look up 'what alpha do I need at
    t=0.5 for this to sound halfway'.

    TODO: implement. Planned: render a coarse alpha grid, measure cumulative
    perceptual arc length, invert that mapping by monotone interpolation,
    refine where spacing error exceeds tolerance.
    """
    raise NotImplementedError("perceptual.uniformity.solve_uniform_schedule")


def apply_schedule(schedule: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Warp curve output through a solved uniformity schedule."""
    if schedule is None:
        return alpha
    grid = np.linspace(0.0, 1.0, len(schedule))
    return np.interp(np.clip(alpha, 0.0, 1.0), grid, schedule)
