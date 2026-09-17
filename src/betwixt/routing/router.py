"""Choose a morph engine by pre-analyzing both files.

Routing table
-------------
both monophonic + pitched      → sinusoidal  (partial-level morph, best quality)
both textural / noise-like     → spectral    (envelope morph)
polyphonic or mixed            → spectral    (codec/diffusion not yet implemented)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.probe import Probe


@dataclass
class Route:
    engine: str
    reason: str
    confidence: float
    rhythm_mode: str


def choose(pa: Probe, pb: Probe, requested: str = "auto",
           rhythm_mode: str = "auto") -> Route:
    """Select an engine and resolve rhythm-mode auto."""
    rhythm = resolve_rhythm_mode(pa, pb, rhythm_mode)

    if requested != "auto":
        return Route(engine=requested, reason="user-specified",
                     confidence=1.0, rhythm_mode=rhythm)

    if pa.is_monophonic_pitched and pb.is_monophonic_pitched:
        engine = "sinusoidal"
        reason = "both monophonic and pitched"
        confidence = 0.85
    elif pa.spectral_flatness > 0.5 and pb.spectral_flatness > 0.5:
        engine = "spectral"
        reason = "both textural/noise-like"
        confidence = 0.8
    else:
        engine = "spectral"
        reason = "general/polyphonic material"
        confidence = 0.6

    return Route(engine=engine, reason=reason, confidence=confidence,
                 rhythm_mode=rhythm)


def resolve_rhythm_mode(pa: Probe, pb: Probe, requested: str) -> str:
    if requested != "auto":
        return requested
    if not pb.is_rhythmic:
        return "dissolve"
    if pa.is_rhythmic or pb.is_rhythmic:
        return "migrate"
    return "hold"
