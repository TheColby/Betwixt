"""Engine interface.

Every engine takes two files plus a per-stream alpha schedule and returns
audio. Engines differ in HOW they represent the signal internally, not in what
they promise: given alpha=0 throughout they must return A, given alpha=1
throughout they must return B, and they must never time-stretch either source.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class MorphRequest:
    a: np.ndarray             # (channels, samples)
    b: np.ndarray
    sample_rate: int
    out_samples: int
    alphas: dict              # stream name -> (frames,) alpha per frame
    rhythm_mode: str
    tail: str
    hold_others: str
    device: str = "cpu"


class MorphEngine(abc.ABC):
    """Base class for all morph engines."""

    name: str = "base"
    #: Streams this engine can control independently. Streams outside this
    #: set are reported to the user as unsupported rather than silently
    #: ignored.
    supports: tuple = ()
    requires_torch: bool = False

    @abc.abstractmethod
    def render(self, req: MorphRequest) -> np.ndarray:
        """Return (channels, out_samples)."""

    def check_endpoints(self, req: MorphRequest, tol: float = 1e-6) -> None:
        """Assert the identity property: alpha=0 reproduces A, alpha=1
        reproduces B. Engines that cannot satisfy this exactly should override
        with a looser perceptual check rather than skipping it."""
        raise NotImplementedError
