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
    # First solve for a perceptual uniformity schedule (unless disabled),
    # then evaluate each stream's curve through that schedule so that
    # 'linear' means perceptually linear.
    # ------------------------------------------------------------------
    hop = 512  # must match SpectralEngine._hop
    n_out_frames = 1 + out_samples // hop

    schedule = None
    if plan.perceptual:
        from .perceptual.uniformity import solve_uniform_schedule, apply_schedule

        def _probe_render(alpha_val: float) -> np.ndarray:
            """Quick low-res render at a single alpha for uniformity probing."""
            probe_alphas = {name: np.array([alpha_val])
                           for name, sp in plan.streams.items() if sp.enabled}
            probe_engine = _get_engine(route.engine)
            # Use a short segment (first 2 seconds) to keep probing fast
            max_probe = min(sr_out * 2, out_samples)
            probe_req = MorphRequest(
                a=xa, b=xb, sample_rate=sr_out, out_samples=max_probe,
                alphas=probe_alphas, rhythm_mode=route.rhythm_mode,
                tail=plan.tail, hold_others=plan.hold_others,
                device=plan.device,
            )
            return probe_engine.render(probe_req)

        if not plan.quiet:
            print("solving perceptual uniformity ...")
        try:
            schedule = solve_uniform_schedule(
                _probe_render, n_points=16, sr=sr_out, device=plan.device)
        except Exception as e:
            import warnings
            warnings.warn(f"perceptual uniformity failed ({e}); using linear",
                          stacklevel=2)
            schedule = None

    alphas: dict[str, np.ndarray] = {}
    t = np.linspace(0.0, 1.0, n_out_frames)
    for name, sp in plan.streams.items():
        if sp.enabled:
            raw = sp.curve(t)
            if schedule is not None:
                from .perceptual.uniformity import apply_schedule
                raw = apply_schedule(schedule, raw)
            alphas[name] = raw

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
    # Step 9: Report — score the morph on perceptual metrics
    # ------------------------------------------------------------------
    if plan.report:
        import json as _json
        from .perceptual.metrics import evaluate as eval_metrics

        if not plan.quiet:
            print("scoring morph quality ...")

        # Render probe points across the morph
        n_probe = 9
        probe_renders = []
        probe_engine = _get_engine(route.engine)
        for i in range(n_probe):
            alpha_val = i / (n_probe - 1)
            probe_alphas = {name: np.array([alpha_val])
                           for name, sp in plan.streams.items() if sp.enabled}
            max_probe = min(sr_out * 2, out_samples)
            probe_req = MorphRequest(
                a=xa, b=xb, sample_rate=sr_out, out_samples=max_probe,
                alphas=probe_alphas, rhythm_mode=route.rhythm_mode,
                tail=plan.tail, hold_others=plan.hold_others,
                device=plan.device,
            )
            probe_renders.append(probe_engine.render(probe_req))

        metrics = eval_metrics(probe_renders, sr_out, plan.device)
        report_path = plan.report
        with open(report_path, "w") as f:
            _json.dump(metrics.as_dict(), f, indent=2)

        if not plan.quiet:
            print(f"report: correspondence={metrics.correspondence:.3f}"
                  f"  smoothness={metrics.smoothness:.3f}"
                  f"  detectability={metrics.detectability:.3f}")
            print(f"wrote {report_path}")

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
