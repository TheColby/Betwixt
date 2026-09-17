"""Choose a morph engine by pre-analyzing both files.

No single technique handles every pair. Monophonic pitched material morphs
best through an explicit sinusoidal model; arbitrary polyphonic material has
no meaningful single f0 and must go through a learned latent space. The
router picks, and --explain shows why.
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
    """Select an engine and resolve rhythm-mode auto.

    Routing table:
      both monophonic + pitched      -> sinusoidal  (exact, cheap, best)
      both textural / noise-like     -> spectral    (envelope morph)
      anything polyphonic or mixed   -> spectral    (envelope morph fallback;
                                                     codec/diffusion not yet impl)
      extreme structural mismatch    -> spectral    (same fallback)

    Rhythm-mode auto resolves against B's pulse salience: migrating a pulse
    requires a pulse to migrate toward, so a rhythmless target degenerates
    to dissolve regardless of what the user asked for.
    """
    rhythm = resolve_rhythm_mode(pa, pb, rhythm_mode)

    if requested != "auto":
        return Route(engine=requested, reason="user-specified",
                     confidence=1.0, rhythm_mode=rhythm)

    if pa.is_monophonic_pitched and pb.is_monophonic_pitched:
        # Sinusoidal engine not yet implemented; fall through to spectral.
        # TODO: re-enable once SinusoidalEngine.render is implemented.
        engine = "spectral"
        reason = "both monophonic/pitched (spectral fallback until sinusoidal lands)"
        confidence = 0.7
    elif pa.spectral_flatness > 0.5 and pb.spectral_flatness > 0.5:
        engine = "spectral"
        reason = "both textural/noise-like"
        confidence = 0.8
    else:
        engine = "spectral"
        reason = "general material"
        confidence = 0.6

    return Route(engine=engine, reason=reason, confidence=confidence,
                 rhythm_mode=rhythm)


def resolve_rhythm_mode(pa: Probe, pb: Probe, requested: str) -> str:
    """Collapse 'auto' to a concrete rhythm mode.

    dissolve  onsets stay where they are; transient salience decays toward B
    migrate   pulse is generatively re-placed toward B's rate
    hold      A's rhythmic grid persists unchanged; only timbre morphs
    """
    if requested != "auto":
        return requested

    if not pb.is_rhythmic:
        return "dissolve"
    if pa.is_rhythmic and pb.is_rhythmic:
        return "migrate"
    if not pa.is_rhythmic and pb.is_rhythmic:
        return "migrate"
    return "hold"
