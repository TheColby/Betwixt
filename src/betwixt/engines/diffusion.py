"""Latent diffusion morphing -- for pairs with no shared structure at all.

The fallback when A and B are so dissimilar that codec-latent interpolation
still produces an audible seam: interpolate in a latent diffusion model's
space and decode, generating plausible intermediate audio rather than
blending two representations.

Expensive, and the most likely to invent material that is in neither source.
Reserved for extreme structural mismatch, and only selected by the router when
the perceptual smoothness metric predicts the codec engine will fail.
"""

from __future__ import annotations

from .base import MorphEngine, MorphRequest


class DiffusionEngine(MorphEngine):
    name = "diffusion"
    supports = ("rhythm", "harmonics", "snr", "centroid", "flatness",
                "loudness")
    requires_torch = True

    def render(self, req: MorphRequest):
        raise NotImplementedError("engines.diffusion.DiffusionEngine.render")
