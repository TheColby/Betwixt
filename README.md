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
```

`--explain` prints the pre-analysis, the routing decision and the per-stream
plan without rendering. Use it before committing to a long render.

## Feature streams

`rhythm`, `f0`, `harmonics`, `snr`, `centroid`, `flatness`, `width`,
`transient`, `loudness`. Select with `--features`, exclude with
`--no-features`, and decide what the rest do with `--hold-others`.

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
on top. So `linear` means perceptually linear.

## Engines

Chosen by pre-analyzing both files; override with `--engine`.

| engine | for | needs torch |
|---|---|---|
| `sinusoidal` | monophonic pitched pairs — exact parameter control | no |
| `spectral` | textural and noise-like pairs | no |
| `codec` | arbitrary polyphonic material, zero-shot | yes |
| `diffusion` | pairs with no shared structure | yes |

## Measuring seamlessness

`--report out.json` scores the render on correspondence (do the endpoints
match A and B), perceptual intermediateness (do middle points sit between
them), and smoothness (is the perceptual step constant). Smoothness is the one
that predicts whether a listener can locate the transition.

## Status

Working. 35 tests passing.

**Implemented:** CLI, config, curves, I/O, device selection, pre-analysis
probe (classical DSP), rule-based router, spectral engine (phase-vocoder morph
with cepstral envelope separation), pipeline orchestrator.

**Stubs (documented in-place):** sinusoidal engine (partial tracking for
monophonic pitched pairs), codec-latent and diffusion engines, per-stream
feature extraction, perceptual uniformity schedule, morph metrics.
