"""Plan construction: CLI and JSON merge."""

import json

import pytest

from betwixt.cli import build_parser, resolve_features
from betwixt.config import MorphPlan, build_plan, resolve_duration


def plan_from(*argv):
    args = build_parser().parse_args(["a.wav", "b.wav", *argv])
    return build_plan(args, resolve_features(args))


def test_global_curve_applies_to_all_streams():
    p = plan_from("--curve", "exp")
    assert all(s.curve.type == "exp" for s in p.streams.values())


def test_json_overrides_global(tmp_path):
    cfg = tmp_path / "m.json"
    cfg.write_text(json.dumps({"f0": {"curve": "log", "rate": 2.0}}))
    p = plan_from("--curve", "exp", "--config", str(cfg))
    assert p.streams["f0"].curve.type == "log"
    assert p.streams["harmonics"].curve.type == "exp"


def test_json_rhythm_mode_wins_over_cli(tmp_path):
    cfg = tmp_path / "m.json"
    cfg.write_text(json.dumps({"rhythm": {"curve": "linear",
                                          "mode": "migrate"}}))
    p = plan_from("--rhythm-mode", "dissolve", "--config", str(cfg))
    assert p.streams["rhythm"].mode == "migrate"


def test_unknown_stream_in_config_rejected(tmp_path):
    cfg = tmp_path / "m.json"
    cfg.write_text(json.dumps({"reverb": {"curve": "linear"}}))
    with pytest.raises(ValueError):
        plan_from("--config", str(cfg))


def test_duration_implies_fixed():
    assert plan_from("--duration", "30").duration_mode == "fixed"


@pytest.mark.parametrize("mode,expected", [
    ("longest", 30.0), ("shortest", 10.0), ("a", 10.0),
    ("b", 30.0), ("sum", 40.0),
])
def test_duration_modes(mode, expected):
    p = plan_from("--duration-mode", mode)
    assert resolve_duration(10.0, 30.0, p) == expected


def test_explain_runs():
    assert "betwixt plan" in plan_from("--curve", "scurve").explain()
