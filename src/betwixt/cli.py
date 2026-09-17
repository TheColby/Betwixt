"""Betwixt command line interface.

Betwixt morphs one audio file into another so that a listener cannot tell
where A stopped and B began. Each audio feature -- rhythm, fundamental,
harmonics, noise floor and more -- travels on its own trajectory, and no
time-stretching is applied to either source.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .curves import CURVE_NAMES

#: Feature streams that can be morphed independently. Selected with
#: --features / --no-features; each accepts its own curve in the JSON config.
FEATURE_STREAMS = (
    "rhythm",       # onset pattern / transient salience
    "f0",           # fundamental frequency contour
    "harmonics",    # harmonic amplitude structure above f0
    "snr",          # harmonic-to-stochastic ratio, noise floor
    "centroid",     # spectral centroid / brightness
    "flatness",     # spectral flatness, tonal vs noise character
    "width",        # stereo width and image
    "transient",    # attack sharpness independent of onset placement
    "loudness",     # perceived level contour
)

DEFAULT_FEATURES = ("rhythm", "f0", "harmonics", "snr", "centroid", "loudness")

RHYTHM_MODES = ("auto", "dissolve", "migrate", "hold")

DURATION_MODES = ("longest", "shortest", "a", "b", "sum", "fixed")

TAIL_MODES = ("hold", "loop", "drop", "fade")

ENGINES = ("auto", "sinusoidal", "spectral", "codec", "diffusion")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="betwixt",
        description="Seamlessly morph one audio file into another, feature by "
                    "feature, with no time-stretching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  betwixt song.wav drone.wav -o out.wav
        default morph: perceptually uniform, longest duration

  betwixt song.wav drone.wav -o out.wav --rhythm-mode dissolve --curve scurve
        beats soften away rather than changing tempo

  betwixt a.wav b.wav -o out.wav --config morph.json
        per-stream curves, modes and windows from JSON

  betwixt a.wav b.wav -o out.wav --features f0,harmonics --hold-others a
        morph only pitch and timbre; everything else stays as A
""")

    p.add_argument("file_a", help="source audio file (any format libsndfile "
                                    "or ffmpeg can read)")
    p.add_argument("file_b", help="target audio file")
    p.add_argument("-o", "--output", default="betwixt.wav",
                    help="output path (default: betwixt.wav)")

    # ---- morph shape -----------------------------------------------------
    g = p.add_argument_group("morph shape")
    g.add_argument("--curve", default="scurve", metavar="NAME",
                    choices=CURVE_NAMES,
                    help=f"global curve applied to every stream lacking its "
                        f"own; one of {', '.join(CURVE_NAMES)} "
                        f"(default: scurve)")
    g.add_argument("--rate", type=float, default=None, metavar="X",
                    help="shape parameter for log/exp curves (default 3.0)")
    g.add_argument("--steepness", type=float, default=None, metavar="X",
                    help="shape parameter for scurve (default 4.0)")
    g.add_argument("--morph-start", type=float, default=0.0, metavar="X",
                    help="morph position at output start (default 0.0)")
    g.add_argument("--morph-end", type=float, default=1.0, metavar="X",
                    help="morph position at output end (default 1.0)")
    g.add_argument("--config", metavar="PATH",
                    help="JSON file giving per-stream curves, modes and "
                        "windows; overrides --curve for streams it names")

    # ---- feature streams -------------------------------------------------
    g = p.add_argument_group("feature streams")
    g.add_argument("--features", metavar="LIST", default=None,
                    help=f"comma-separated streams to morph, or 'all'. "
                        f"available: {', '.join(FEATURE_STREAMS)} "
                        f"(default: {','.join(DEFAULT_FEATURES)})")
    g.add_argument("--no-features", metavar="LIST", default=None,
                    help="comma-separated streams to exclude from the "
                        "selection")
    g.add_argument("--hold-others", choices=("a", "b", "blend"), default="a",
                    help="what unselected streams do: follow A, follow B, or "
                        "sit at a static blend (default: a)")

    # ---- rhythm ----------------------------------------------------------
    g = p.add_argument_group("rhythm")
    g.add_argument("--rhythm-mode", choices=RHYTHM_MODES, default="auto",
                    help="dissolve: onsets stay put, transient salience "
                        "decays. migrate: pulse is generatively re-placed "
                        "toward B's rate. hold: A's grid persists. auto: "
                        "decide from B's pulse salience (default: auto)")

    # ---- duration --------------------------------------------------------
    g = p.add_argument_group("duration")
    g.add_argument("--duration-mode", choices=DURATION_MODES,
                    default="longest",
                    help="output length: longest/shortest of the two inputs, "
                        "a, b, sum, or fixed with --duration "
                        "(default: longest)")
    g.add_argument("--duration", type=float, default=None, metavar="SEC",
                    help="explicit output length in seconds; implies "
                        "--duration-mode fixed")
    g.add_argument("--tail", choices=TAIL_MODES, default="hold",
                    help="what the shorter input does once exhausted: hold "
                        "its final state, loop, drop out, or fade "
                        "(default: hold)")

    # ---- engine ----------------------------------------------------------
    g = p.add_argument_group("engine")
    g.add_argument("--engine", choices=ENGINES, default="auto",
                    help="morph engine; auto pre-analyzes both files and "
                        "routes (default: auto)")
    g.add_argument("--no-perceptual", action="store_true",
                    help="skip perceptual reparametrization and interpolate "
                        "linearly in parameter space (diagnostic)")
    g.add_argument("--device", default="auto", metavar="DEV",
                    help="auto, cpu, cuda, cuda:N, or mps (default: auto)")
    g.add_argument("--explain", action="store_true",
                    help="print pre-analysis, routing decision and per-stream "
                        "plan, then exit without rendering")

    # ---- output ----------------------------------------------------------
    g = p.add_argument_group("output")
    g.add_argument("--sr", type=int, default=None, metavar="HZ",
                    help="output sample rate (default: highest input rate)")
    g.add_argument("--mono", action="store_true", help="force mono output")
    g.add_argument("--peak", type=float, default=-1.0, metavar="DBFS",
                    help="normalize to this peak in dBFS; use --no-normalize "
                        "to disable (default: -1.0)")
    g.add_argument("--no-normalize", action="store_true")
    g.add_argument("--report", metavar="PATH",
                    help="write a JSON report with morph metrics "
                        "(correspondence, intermediateness, smoothness)")
    g.add_argument("-v", "--verbose", action="count", default=0)
    g.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--version", action="version",
                    version=f"betwixt {__version__}")
    return p


def resolve_features(args) -> tuple:
    """Apply --features / --no-features to produce the active stream set."""
    if args.features in (None, ""):
        selected = list(DEFAULT_FEATURES)
    elif args.features.strip().lower() == "all":
        selected = list(FEATURE_STREAMS)
    else:
        selected = [s.strip() for s in args.features.split(",") if s.strip()]

    unknown = [s for s in selected if s not in FEATURE_STREAMS]
    if unknown:
        raise ValueError(
            f"unknown feature stream(s): {', '.join(unknown)}. "
            f"available: {', '.join(FEATURE_STREAMS)}")

    if args.no_features:
        drop = {s.strip() for s in args.no_features.split(",") if s.strip()}
        unknown = [s for s in drop if s not in FEATURE_STREAMS]
        if unknown:
            raise ValueError(
                f"unknown feature stream(s) in --no-features: "
                f"{', '.join(unknown)}")
        selected = [s for s in selected if s not in drop]

    if not selected:
        raise ValueError("no feature streams selected; nothing to morph")
    return tuple(selected)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from .config import build_plan
    from .device import select_device
    from .pipeline import render

    try:
        features = resolve_features(args)
        plan = build_plan(args, features)
    except (ValueError, OSError) as e:
        print(f"betwixt: {e}", file=sys.stderr)
        return 2

    plan.device = select_device(args.device)

    if args.explain:
        print(plan.explain())
        return 0

    try:
        render(plan)
    except NotImplementedError as e:
        print(f"betwixt: not implemented yet: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"betwixt: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
