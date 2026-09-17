"""Render orchestration: plan in, audio out.

Order matters here. Pre-analysis precedes routing because the engine choice
depends on material. Perceptual reparametrization precedes curve evaluation
because the user's curve is a warp applied to an already-uniform axis, not a
substitute for one.
"""

from __future__ import annotations

import numpy as np

from .config import MorphPlan, resolve_duration


def render(plan: MorphPlan) -> np.ndarray:
    """Execute a MorphPlan end to end.

    Steps:
      1. load A and B, reconcile sample rates and channel counts
      2. probe both files
      3. route to an engine, resolve rhythm-mode auto
      4. extract the selected feature streams from both
      5. solve the perceptual uniformity schedule (unless --no-perceptual)
      6. evaluate each stream's curve over the output frame grid, warped
         through that schedule
      7. render with the chosen engine
      8. apply tail policy, normalize, write
      9. if --report, score with perceptual.metrics and write JSON
    """
    from .io import load, save, normalize_peak, resample
    from .analysis.probe import probe
    from .routing.router import choose
    from .engines.base import MorphRequest

    # ------------------------------------------------------------------
    # Step 1: Load A and B, reconcile sample rates and channel counts
    # ------------------------------------------------------------------
    if not plan.quiet:
        print(f"loading {plan.file_a}")
    xa, sr_a = load(plan.file_a, mono=plan.mono)

    if not plan.quiet:
        print(f"loading {plan.file_b}")
    xb, sr_b = load(plan.file_b, mono=plan.mono)

    sr_out = plan.sr or max(sr_a, sr_b)
    if sr_a != sr_out:
        xa = resample(xa, sr_a, sr_out)
    if sr_b != sr_out:
        xb = resample(xb, sr_b, sr_out)

    n_ch = max(xa.shape[0], xb.shape[0])
    if xa.shape[0] < n_ch:
        xa = np.concatenate([xa] * n_ch, axis=0)[:n_ch]
    if xb.shape[0] < n_ch:
        xb = np.concatenate([xb] * n_ch, axis=0)[:n_ch]

    # ------------------------------------------------------------------
    # Step 2: Probe both files
    # ------------------------------------------------------------------
    if plan.verbose >= 1 and not plan.quiet:
        print("probing ...")
    pa = probe(xa, sr_out)
    pb = probe(xb, sr_out)

    if plan.verbose >= 2 and not plan.quiet:
        print(f"  A: harmonicity={pa.harmonicity:.2f}  pulse={pa.pulse_salience:.2f}"
              f"  flatness={pa.spectral_flatness:.2f}  poly={pa.polyphony:.1f}")
        print(f"  B: harmonicity={pb.harmonicity:.2f}  pulse={pb.pulse_salience:.2f}"
              f"  flatness={pb.spectral_flatness:.2f}  poly={pb.polyphony:.1f}")

    # ------------------------------------------------------------------
    # Step 3: Route to an engine; resolve rhythm-mode auto
    # ------------------------------------------------------------------
    rhythm_sp = plan.streams.get("rhythm")
    requested_rhythm = rhythm_sp.mode if rhythm_sp else "auto"
    route = choose(pa, pb, requested=plan.engine, rhythm_mode=requested_rhythm)

    if not plan.quiet:
        print(f"engine: {route.engine}  ({route.reason})"
              f"  rhythm: {route.rhythm_mode}")

    dur_out = resolve_duration(pa.duration, pb.duration, plan)
    out_samples = max(1, int(round(dur_out * sr_out)))

    # ------------------------------------------------------------------
    # Steps 4–6: Build per-stream alpha schedules
    #
    # Feature extraction and perceptual uniformity are not yet implemented.
    # The pipeline falls back to evaluating each stream's curve directly on
    # a linear time axis (equivalent to --no-perceptual).  When
    # perceptual.uniformity.solve_uniform_schedule is implemented, the
    # solved schedule is interposed here as a warp of the t axis before
    # curve evaluation.
    # ------------------------------------------------------------------
    hop = 512  # must match SpectralEngine._hop
    n_out_frames = 1 + out_samples // hop

    alphas: dict[str, np.ndarray] = {}
    t = np.linspace(0.0, 1.0, n_out_frames)
    for name, sp in plan.streams.items():
        if sp.enabled:
            alphas[name] = sp.curve(t)

    # ------------------------------------------------------------------
    # Step 7: Render
    # ------------------------------------------------------------------
    engine = _get_engine(route.engine)

    req = MorphRequest(
        a=xa,
        b=xb,
        sample_rate=sr_out,
        out_samples=out_samples,
        alphas=alphas,
        rhythm_mode=route.rhythm_mode,
        tail=plan.tail,
        hold_others=plan.hold_others,
        device=plan.device,
    )

    if not plan.quiet:
        print("rendering ...")
    y = engine.render(req)

    # ------------------------------------------------------------------
    # Step 8: Normalize and write
    # ------------------------------------------------------------------
    if plan.peak_dbfs is not None:
        y = normalize_peak(y, plan.peak_dbfs)

    save(plan.output, y, sr_out)

    if not plan.quiet:
        print(f"wrote {plan.output}")

    # ------------------------------------------------------------------
    # Step 9: Report (not yet implemented)
    # ------------------------------------------------------------------
    if plan.report:
        import warnings
        warnings.warn("--report is not yet implemented; skipping", stacklevel=2)

    return y


def _get_engine(name: str):
    from .engines.spectral import SpectralEngine
    from .engines.sinusoidal import SinusoidalEngine

    registry = {
        "spectral": SpectralEngine,
        "sinusoidal": SinusoidalEngine,
    }
    cls = registry.get(name)
    if cls is None:
        import warnings
        warnings.warn(
            f"engine '{name}' is not yet implemented; falling back to spectral",
            stacklevel=3)
        cls = SpectralEngine
    return cls()
