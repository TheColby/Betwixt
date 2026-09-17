"""Sinusoidal-model morphing for monophonic pitched material.

Partial tracking on both sources, partial-to-partial matching, then
independent interpolation of each matched partial's frequency and amplitude,
plus a separately morphed stochastic residual. This is the classical answer
to timbre morphing and gives genuinely clean intermediate sounds -- but it
assumes a single f0, so the router only selects it for monophonic pairs.

Gives the finest-grained control of any engine: f0, harmonics and snr map
directly onto model parameters rather than being inferred.
"""

from __future__ import annotations

from .base import MorphEngine, MorphRequest


class SinusoidalEngine(MorphEngine):
    name = "sinusoidal"
    supports = ("f0", "harmonics", "snr", "centroid", "transient", "loudness")
    requires_torch = False

    def render(self, req: MorphRequest):
        raise NotImplementedError(
            "engines.sinusoidal.SinusoidalEngine.render")
