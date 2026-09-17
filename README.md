# Betwixt

Seamlessly morph one audio file into another. No time-stretching. Every
feature — rhythm, fundamental, harmonics, noise floor, brightness, width —
travels on its own trajectory, so you can dissolve a pop song's beat away
early while its pitch holds until the very end.

The goal is a single criterion: **a listener cannot tell where A stopped and
B began.**

## Install

```bash
pip install -e .              # classical DSP engines only
pip install -e ".[neural]"    # adds codec-latent and diffusion engines
```

FFmpeg is optional; it is used automatically for any format libsndfile can't
open (mp3, m4a, opus, video containers).

## Use

```bash
betwixt song.wav drone.wav -o out.wav
betwixt song.wav drone.wav -o out.wav --rhythm-mode dissolve --curve scurve
betwixt a.wav b.wav -o out.wav --config examples/morph.json
betwixt a.wav b.wav --explain            # show the plan, render nothing
betwixt a.wav b.wav -o out.wav --report scores.json  # render + quality report
```

`--explain` prints the pre-analysis, the routing decision and the per-stream
plan without rendering. Use it before committing to a long render.

## Feature streams

`rhythm`, `f0`, `harmonics`, `snr`, `centroid`, `flatness`, `width`,
`transient`, `loudness`. Select with `--features`, exclude with
`--no-features`, and decide what the rest do with `--hold-others`.

Each stream is extracted independently: onset envelopes for rhythm, pYIN for
f0, HPSS energy ratio for snr, spectral centroid in Hz, spectral flatness,
stereo mid/side ratio for width, attack slope for transients, and RMS loudness
in dB.

## Rhythm

Because nothing is time-stretched, "rhythm morphing" needs a stated meaning:

| mode | behavior |
|---|---|
| `dissolve` | onsets stay where they are; transient salience decays toward B |
| `migrate` | the pulse is generatively re-placed toward B's rate |
| `hold` | A's rhythmic grid persists; only timbre morphs |
| `auto` | resolved from B's pulse salience |

`auto` matters because migrating a pulse requires a pulse to migrate toward.
Against a drone, `migrate` has no target and degenerates to `dissolve`.

## Curves

`linear`, `log`, `exp`, `scurve`, `cosine`, `hold-a`, `hold-b`, `step`.

Set one globally with `--curve`, or give each stream its own in JSON,
including a time window so streams can start and finish at different points:

```json
{
  "default":   { "curve": "scurve", "steepness": 4.0 },
  "rhythm":    { "mode": "dissolve", "curve": "exp", "rate": 2.5,
                 "start": 0.0, "end": 0.7 },
  "f0":        { "curve": "linear", "start": 0.3, "end": 1.0 },
  "snr":       { "curve": "log", "rate": 2.0 }
}
```

**Curves are applied in perceptual space.** Linear interpolation of a morph
factor does not produce linear perceptual change — it sounds like one endpoint
for most of the run, then lurches through the middle, and that lurch is where
a listener notices the seam. Betwixt first solves for a trajectory whose
intermediate results are perceptually equally spaced, then applies your curve
on top. So `linear` means perceptually linear. Disable with `--no-perceptual`.

## Perceptual uniformity

Betwixt probes the morph at multiple alpha positions, measures log-Mel spectral
distance between consecutive renders, computes cumulative perceptual arc length,
and inverts that mapping so that the final alpha schedule produces renders that
are equally spaced in perceptual distance. The solver runs iteratively until
spacing error drops below tolerance. This is the key difference between a morph
that sounds like a crossfade and one that sounds like a smooth transformation.

## Engines

Chosen by pre-analyzing both files; override with `--engine`.

| engine | for | needs torch |
|---|---|---|
| `sinusoidal` | monophonic pitched pairs — partial-level frequency interpolation with Hungarian matching | no |
| `spectral` | textural, noise-like, and general polyphonic material — HPSS + PV + cepstral envelope | no |
| `codec` | arbitrary polyphonic material, zero-shot | yes |
| `diffusion` | pairs with no shared structure | yes |

### Spectral engine

Separates harmonic and percussive components via HPSS before morphing. The
harmonic component gets full phase-vocoder treatment: instantaneous frequency
extraction, cepstral envelope/fine-structure separation (so formants shift
independently of pitch), and phase-coherent synthesis via IF integration. The
percussive component is magnitude-blended with phase selection from the
dominant source. The `centroid` stream controls formant morph rate; `harmonics`
controls fine-structure morph rate; `snr` controls the percussive blend.

### Sinusoidal engine

For monophonic pitched material. Detects spectral peaks in both sources,
matches them across A and B using the Hungarian algorithm (minimizing frequency
distance), then interpolates matched partials in frequency and amplitude with
phase-coherent accumulation. Unmatched A partials fade out by (1-alpha);
unmatched B partials fade in by alpha. Non-peak bins are blended as stochastic
residual.

## Formant morphing

The spectral engine separates the spectral envelope (formants) from harmonic
fine structure using cepstral liftering. The `centroid` stream controls how
fast formants shift from A to B. The `harmonics` stream controls pitch/harmonic
content independently. To morph formants first and pitch later:

```json
{
  "centroid":  { "curve": "linear", "start": 0.0, "end": 0.5 },
  "harmonics": { "curve": "linear", "start": 0.5, "end": 1.0 }
}
```

## Measuring seamlessness

`--report out.json` scores the render on correspondence (do the endpoints
match A and B), perceptual intermediateness (do middle points sit between
them), and smoothness (is the perceptual step constant). Smoothness is the one
that predicts whether a listener can locate the transition. Detectability
estimates how likely a listener is to notice the seam based on the worst-case
step size.

## Pre-analysis

Both files are probed for harmonicity (FFT autocorrelation), pulse salience
(onset-envelope autocorrelation), spectral flatness (geometric/arithmetic mean),
onset density (spectral flux), and polyphony. These drive automatic engine
selection and rhythm-mode resolution.

## Status

Working. 35 tests passing.

**Implemented:** CLI, config, curves, I/O, device selection, pre-analysis
probe, rule-based router, spectral engine (HPSS separation + phase-vocoder
morph with cepstral envelope/fine-structure independence), sinusoidal engine
(Hungarian partial matching + phase-coherent synthesis), perceptual uniformity
solver (log-Mel arc-length inversion), feature extraction (all 9 streams),
morph quality metrics (correspondence, intermediateness, smoothness,
detectability), `--report` scoring, pipeline orchestrator.

**Stubs:** codec-latent and diffusion engines.
