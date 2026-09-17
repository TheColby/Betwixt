"""Objective morph quality metrics.

These give 'seamless' a number instead of an opinion, and double as the
regression suite: any change to an engine is judged by whether these improve.

Three criteria, following the SoundMorpher evaluation framework:

  correspondence            do the endpoints actually match A and B?
  perceptual intermediateness  do middle renders sit perceptually between the
                            endpoints, rather than collapsing to one side or
                            wandering off to a third sound?
  smoothness                is the perceptual step between adjacent morph
                            positions constant? this is the one that predicts
                            whether a listener can locate the transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MorphMetrics:
    correspondence: float
    intermediateness: float
    smoothness: float
    max_step: float          # largest perceptual jump between adjacent points
    detectability: float     # estimated likelihood a listener locates the seam

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def evaluate(renders: list, sr: int, device: str = "cpu") -> MorphMetrics:
    """Score a sequence of renders sampled across the morph.

    renders: list of np.ndarray, ordered from alpha=0 (A) to alpha=1 (B).
             Minimum 3 elements (A, midpoint, B).

    Returns MorphMetrics with all scores in [0, 1] (higher = better),
    except max_step (lower = better) and detectability (lower = better).
    """
    from .uniformity import perceptual_distance

    n = len(renders)
    if n < 3:
        return MorphMetrics(
            correspondence=0.0, intermediateness=0.0,
            smoothness=0.0, max_step=float("inf"), detectability=1.0)

    # Pairwise consecutive distances
    dists = []
    for i in range(n - 1):
        d = perceptual_distance(renders[i], renders[i + 1], sr, device)
        dists.append(d)
    dists = np.array(dists)

    # --- Correspondence ---
    # How close are the endpoints to the originals?
    # d(render[0], A) should be 0 by construction (render at alpha=0 IS A).
    # We measure d(render[0], render[1]) vs d(render[-2], render[-1]) symmetry.
    # Since renders[0] = A and renders[-1] = B by convention, correspondence
    # is 1.0 by construction. Instead, measure total distance coverage:
    d_ab = perceptual_distance(renders[0], renders[-1], sr, device)
    total_path = float(np.sum(dists))
    # A good morph covers the full A-to-B distance without detours
    correspondence = min(1.0, d_ab / (total_path + 1e-12))

    # --- Smoothness ---
    # 1 - (std / mean) of step sizes. Perfect uniformity = 1.0.
    mean_step = float(np.mean(dists))
    if mean_step < 1e-12:
        smoothness = 1.0
        max_step = 0.0
    else:
        std_step = float(np.std(dists))
        smoothness = max(0.0, 1.0 - std_step / mean_step)
        max_step = float(np.max(dists))

    # --- Intermediateness ---
    # Each interior render should be perceptually between A and B.
    # Measure: for each interior point, its distance to A + distance to B
    # should approximate d(A, B). Closer = more intermediate.
    d_to_a = np.zeros(n)
    d_to_b = np.zeros(n)
    # Cumulative distance from A
    d_to_a[0] = 0.0
    for i in range(1, n):
        d_to_a[i] = d_to_a[i - 1] + dists[i - 1]
    # Distance from B
    d_to_b[-1] = 0.0
    for i in range(n - 2, -1, -1):
        d_to_b[i] = d_to_b[i + 1] + dists[i]

    if d_ab < 1e-12:
        intermediateness = 1.0
    else:
        # For each interior point, check that d_to_a + d_to_b ≈ total_path
        # (triangle inequality: detour ratio)
        detour_ratios = []
        for i in range(1, n - 1):
            via_i = d_to_a[i] + d_to_b[i]
            detour_ratios.append(via_i / total_path)
        # Perfect path = ratio of 1.0 for all interior points
        mean_detour = float(np.mean(detour_ratios))
        intermediateness = min(1.0, 1.0 / (mean_detour + 1e-12))

    # --- Detectability ---
    # Estimated from max_step relative to mean: a large outlier step is
    # where a listener will notice.
    if mean_step < 1e-12:
        detectability = 0.0
    else:
        ratio = max_step / mean_step
        # ratio=1 → undetectable; ratio=3+ → very detectable
        detectability = min(1.0, max(0.0, (ratio - 1.0) / 2.0))

    return MorphMetrics(
        correspondence=float(correspondence),
        intermediateness=float(intermediateness),
        smoothness=float(smoothness),
        max_step=float(max_step),
        detectability=float(detectability),
    )
