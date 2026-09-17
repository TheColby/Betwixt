"""Curve behavior. These are properties every curve must satisfy for the
morph to be seamless -- endpoints exact, monotone, in range."""

import numpy as np
import pytest

from betwixt.curves import CURVE_NAMES, Curve, evaluate

SHAPED = [n for n in CURVE_NAMES if n != "step"]


@pytest.mark.parametrize("name", SHAPED)
def test_endpoints_exact(name):
    c = Curve(type=name)
    out = c(np.array([0.0, 1.0]))
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", SHAPED)
def test_monotone_and_bounded(name):
    out = evaluate(Curve(type=name), 256)
    assert np.all(np.diff(out) >= -1e-12), f"{name} is not monotone"
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_window_delays_stream():
    c = Curve(type="linear", start=0.5, end=1.0)
    t = np.linspace(0, 1, 101)
    out = c(t)
    assert np.all(out[:50] == 0.0)
    assert out[-1] == pytest.approx(1.0)


def test_from_spec_string_and_dict():
    assert Curve.from_spec("exp").type == "exp"
    c = Curve.from_spec({"curve": "scurve", "steepness": 6.0, "mode": "hold"})
    assert c.type == "scurve" and c.params["steepness"] == 6.0


def test_unknown_curve_rejected():
    with pytest.raises(ValueError):
        Curve(type="parabolic")


def test_bad_window_rejected():
    with pytest.raises(ValueError):
        Curve(type="linear", start=0.8, end=0.2)
