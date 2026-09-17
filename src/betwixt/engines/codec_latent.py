"""Neural audio codec latent morphing -- the engine for arbitrary material.

Zero-shot, no per-pair training, and indifferent to polyphony, which is what
the pop-song-into-drone case requires.

Why this works without training: operating in the latent space of a pretrained
neural audio codec removes the need for corpus-specific training while giving
representations that interpolate naturally during decoding, and the decoder's
final upsampling implicitly smooths between grains, avoiding the
discontinuities that plague concatenative approaches.

Stream control maps onto RVQ codebook structure. Residual vector quantization
stages capture progressively finer detail, and recent training-free work
separates coarse, middle and fine codebook groups as distinct transfer
targets, paired with a continuity-constrained sequence matcher (bounded beam
search) rather than independent greedy selection. Roughly:

    coarse codebooks  -> structure and rhythmic organization
    middle codebooks  -> timbre and harmonic character
    fine codebooks    -> noise floor and fine detail

That mapping is the lever for per-stream curves here: rhythm rides the coarse
group, harmonics the middle, snr the fine. It is coarser than the explicit
parameter control the sinusoidal engine offers, which is the price of working
on arbitrary audio.

Candidate backbones (all pretrained, weights fetched on first use):
    DAC (Improved RVQGAN), EnCodec, DisCodec (music-specific)

References:
    Tokui et al., "Latent Granular Resynthesis using Neural Audio Codecs"
        (arXiv:2507.19202)
    "Neural Morphing: Sequence-Optimized Token-Level Morphing in Neural Audio
        Codecs" (arXiv:2607.12725)
"""

from __future__ import annotations

from .base import MorphEngine, MorphRequest


class CodecLatentEngine(MorphEngine):
    name = "codec"
    supports = ("rhythm", "f0", "harmonics", "snr", "centroid", "flatness",
                "width", "transient", "loudness")
    requires_torch = True

    def __init__(self, backbone: str = "dac", device: str = "cpu"):
        self.backbone = backbone
        self.device = device

    def render(self, req: MorphRequest):
        raise NotImplementedError(
            "engines.codec_latent.CodecLatentEngine.render")
