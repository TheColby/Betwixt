"""Morph trajectory curves.

A curve maps normalized time t in [0, 1] to a morph position alpha in [0, 1].
Every feature stream carries its own curve, so rhythm can dissolve early while
the fundamental holds until late.

IMPORTANT: curves are applied in PERCEPTUAL space, not parameter space. The
perceptual.uniformity module first reparametrizes the raw morph axis so that
equal steps in alpha produce equal perceptual change; the curve then warps that
already-uniform axis for artistic effect. A "linear" curve therefore means
perceptually linear, which is what a listener actually experiences as even.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np

CurveFn = Callable[[np.ndarray], np.ndarray]

#: Curve names accepted by --curve and by the JSON config.
CURVE_NAMES = (
    "linear",    # constant rate
    "log",       # fast early, slow late
    "exp",       # slow early, fast late
    "scurve",    # ease in and out; most "invisible" for long morphs
    "cosine",    # equal-power style, gentler than scurve
    "hold-a",    # stay at A, then collapse to B at the very end
    "hold-b",    # jump to B immediately, then hold
    "step",      # hard cut at --at (diagnostic / A-B reference)
)


def _clamp01(t: np.ndarray) -> np.ndarray:
    return np.clip(t, 0.0, 1.0)


def linear(t: np.ndarray, **_) -> np.ndarray:
    return _clamp01(t)


def exp_curve(t: np.ndarray, rate: float = 3.0, **_) -> np.ndarray:
    """Slow departure from A, accelerating into B."""
    t = _clamp01(t)
    if abs(rate) < 1e-9:
        return t
    return (np.exp(rate * t) - 1.0) / (np.exp(rate) - 1.0)


def log_curve(t: np.ndarray, rate: float = 3.0, **_) -> np.ndarray:
    """Fast departure from A, long approach to B. Inverse of exp_curve."""
    t = _clamp01(t)
    if abs(rate) < 1e-9:
        return t
    return np.log1p(t * (np.exp(rate) - 1.0)) / rate


def scurve(t: np.ndarray, steepness: float = 4.0, **_) -> np.ndarray:
    """Logistic ease-in/ease-out, renormalized to hit 0 and 1 exactly."""
    t = _clamp01(t)
    if abs(steepness) < 1e-9:
        return t
    raw = 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))
    lo = 1.0 / (1.0 + np.exp(steepness * 0.5))
    hi = 1.0 / (1.0 + np.exp(-steepness * 0.5))
    return (raw - lo) / (hi - lo)


def cosine(t: np.ndarray, **_) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * _clamp01(t))


def hold_a(t: np.ndarray, knee: float = 0.8, **_) -> np.ndarray:
    """Remain at A until `knee`, then travel to B over the remainder."""
    t = _clamp01(t)
    knee = float(np.clip(knee, 0.0, 0.999))
    return _clamp01((t - knee) / (1.0 - knee))


def hold_b(t: np.ndarray, knee: float = 0.2, **_) -> np.ndarray:
    """Reach B by `knee`, then hold."""
    t = _clamp01(t)
    knee = float(np.clip(knee, 0.001, 1.0))
    return _clamp01(t / knee)


def step(t: np.ndarray, at: float = 0.5, **_) -> np.ndarray:
    """Hard cut. Not a morph -- kept as the reference stimulus that Betwixt
    is trying to be perceptually distinguishable from."""
    return (_clamp01(t) >= at).astype(float)


_REGISTRY: Dict[str, CurveFn] = {
    "linear": linear,
    "log": log_curve,
    "exp": exp_curve,
    "scurve": scurve,
    "cosine": cosine,
    "hold-a": hold_a,
    "hold-b": hold_b,
    "step": step,
}


@dataclass
class Curve:
    """A named curve plus its shape parameters and optional time window.

    start/end delay or truncate a stream's morph relative to the global
    timeline, so (for example) rhythm can begin dissolving at 20% while the
    fundamental does not move until 60%.
    """

    type: str = "linear"
    params: Dict[str, float] = field(default_factory=dict)
    start: float = 0.0
    end: float = 1.0

    def __post_init__(self):
        if self.type not in _REGISTRY:
            raise ValueError(
                f"unknown curve '{self.type}'; expected one of "
                f"{', '.join(CURVE_NAMES)}")
        if not 0.0 <= self.start < self.end <= 1.0:
            raise ValueError(
                f"curve window must satisfy 0 <= start < end <= 1 "
                f"(got start={self.start}, end={self.end})")

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Evaluate over normalized time, honoring the stream's window."""
        t = np.asarray(t, dtype=float)
        local = (t - self.start) / (self.end - self.start)
        return _clamp01(_REGISTRY[self.type](_clamp01(local), **self.params))

    @classmethod
    def from_spec(cls, spec) -> "Curve":
        """Build from a string ('exp') or a dict ({'curve': 'exp', ...})."""
        if isinstance(spec, str):
            return cls(type=spec)
        if not isinstance(spec, dict):
            raise TypeError(f"cannot read curve spec from {type(spec)}")
        spec = dict(spec)
        ctype = spec.pop("curve", spec.pop("type", "linear"))
        start = float(spec.pop("start", 0.0))
        end = float(spec.pop("end", 1.0))
        spec.pop("mode", None)  # belongs to the stream, not the curve
        params = {k: float(v) for k, v in spec.items()}
        return cls(type=ctype, params=params, start=start, end=end)


def evaluate(curve: Curve, n_frames: int) -> np.ndarray:
    """Sample a curve across n_frames, returning alpha per frame."""
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    if n_frames == 1:
        return curve(np.array([0.0]))
    return curve(np.linspace(0.0, 1.0, n_frames))
