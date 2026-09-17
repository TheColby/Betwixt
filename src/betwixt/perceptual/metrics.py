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

    TODO: implement. smoothness is the key one: compute perceptual distance
    between consecutive renders and return 1 - (std / mean), so a perfectly
    even trajectory scores 1.0. max_step localizes the worst seam.
    """
    raise NotImplementedError("perceptual.metrics.evaluate")
