"""Morph plan: the fully-resolved description of one render.

CLI switches and the JSON config are merged here into a single MorphPlan so
that no downstream module has to know where a setting came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from .curves import Curve


@dataclass
class StreamPlan:
    """How one feature stream behaves across the morph."""
    name: str
    curve: Curve
    mode: str | None = None       # rhythm uses this; other streams ignore it
    enabled: bool = True


@dataclass
class MorphPlan:
    file_a: str
    file_b: str
    output: str
    streams: Dict[str, StreamPlan]
    hold_others: str = "a"
    duration_mode: str = "longest"
    duration: float | None = None
    tail: str = "hold"
    engine: str = "auto"
    perceptual: bool = True
    morph_start: float = 0.0
    morph_end: float = 1.0
    device: str = "cpu"
    sr: int | None = None
    mono: bool = False
    peak_dbfs: float | None = -1.0
    report: str | None = None
    verbose: int = 0
    quiet: bool = False
    meta: dict = field(default_factory=dict)

    def explain(self) -> str:
        from .device import describe
        lines = [
            "betwixt plan",
            f"  A            {self.file_a}",
            f"  B            {self.file_b}",
            f"  output       {self.output}",
            f"  engine       {self.engine}",
            f"  device       {describe(self.device)}",
            f"  duration     {self.duration_mode}"
            + (f" ({self.duration}s)" if self.duration else "")
            + f", tail={self.tail}",
            f"  perceptual   {'on' if self.perceptual else 'off (linear)'}",
            f"  morph range  {self.morph_start} -> {self.morph_end}",
            f"  unselected   follow {self.hold_others}",
            "  streams",
        ]
        for name, sp in self.streams.items():
            if not sp.enabled:
                continue
            win = ("" if (sp.curve.start, sp.curve.end) == (0.0, 1.0)
                   else f"  window {sp.curve.start}-{sp.curve.end}")
            mode = f"  mode={sp.mode}" if sp.mode else ""
            params = (f"  {sp.curve.params}" if sp.curve.params else "")
            lines.append(f"    {name:<10} {sp.curve.type}{params}{mode}{win}")
        return "\n".join(lines)


def _global_params(args) -> dict:
    params = {}
    if args.rate is not None:
        params["rate"] = args.rate
    if args.steepness is not None:
        params["steepness"] = args.steepness
    return params


def build_plan(args, features: tuple) -> MorphPlan:
    """Merge CLI arguments with an optional JSON config into a MorphPlan."""
    cfg = {}
    if args.config:
        path = Path(args.config)
        if not path.exists():
            raise OSError(f"config not found: {path}")
        try:
            cfg = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON in {path}: {e}") from e
        if not isinstance(cfg, dict):
            raise ValueError(f"{path}: top level must be an object")

    from .cli import FEATURE_STREAMS
    unknown = [k for k in cfg if k not in FEATURE_STREAMS and k != "default"]
    if unknown:
        raise ValueError(
            f"unknown stream(s) in config: {', '.join(unknown)}")

    default_spec = cfg.get("default")
    gparams = _global_params(args)

    streams: Dict[str, StreamPlan] = {}
    for name in FEATURE_STREAMS:
        spec = cfg.get(name, default_spec)
        if spec is None:
            curve = Curve(type=args.curve, params=dict(gparams))
        else:
            curve = Curve.from_spec(spec)
            for k, v in gparams.items():
                curve.params.setdefault(k, v)

        mode = None
        if name == "rhythm":
            mode = (spec.get("mode") if isinstance(spec, dict) else None) \
                   or args.rhythm_mode
        streams[name] = StreamPlan(name=name, curve=curve, mode=mode,
                                   enabled=name in features)

    duration_mode = args.duration_mode
    if args.duration is not None:
        duration_mode = "fixed"
    if duration_mode == "fixed" and args.duration is None:
        raise ValueError("--duration-mode fixed requires --duration SEC")

    for label, val in (("--morph-start", args.morph_start),
                       ("--morph-end", args.morph_end)):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{label} must be between 0 and 1 (got {val})")

    return MorphPlan(
        file_a=args.file_a,
        file_b=args.file_b,
        output=args.output,
        streams=streams,
        hold_others=args.hold_others,
        duration_mode=duration_mode,
        duration=args.duration,
        tail=args.tail,
        engine=args.engine,
        perceptual=not args.no_perceptual,
        morph_start=args.morph_start,
        morph_end=args.morph_end,
        sr=args.sr,
        mono=args.mono,
        peak_dbfs=None if args.no_normalize else args.peak,
        report=args.report,
        verbose=args.verbose,
        quiet=args.quiet,
    )


def resolve_duration(dur_a: float, dur_b: float, plan: MorphPlan) -> float:
    """Output length in seconds. No time-stretching is ever applied; this
    only decides how long the canvas is."""
    mode = plan.duration_mode
    if mode == "longest":
        return max(dur_a, dur_b)
    if mode == "shortest":
        return min(dur_a, dur_b)
    if mode == "a":
        return dur_a
    if mode == "b":
        return dur_b
    if mode == "sum":
        return dur_a + dur_b
    if mode == "fixed":
        return float(plan.duration)
    raise ValueError(f"unknown duration mode: {mode}")
