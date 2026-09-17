"""CLI surface: feature selection and plan construction."""

import pytest

from betwixt.cli import DEFAULT_FEATURES, FEATURE_STREAMS, build_parser, \
    resolve_features


def parse(*argv):
    return build_parser().parse_args(["a.wav", "b.wav", *argv])


def test_default_features():
    assert resolve_features(parse()) == DEFAULT_FEATURES


def test_all_features():
    assert resolve_features(parse("--features", "all")) == FEATURE_STREAMS


def test_subtract_features():
    got = resolve_features(parse("--features", "all", "--no-features", "width"))
    assert "width" not in got and "f0" in got


def test_unknown_feature_rejected():
    with pytest.raises(ValueError):
        resolve_features(parse("--features", "reverb"))


def test_empty_selection_rejected():
    with pytest.raises(ValueError):
        resolve_features(parse("--features", "f0", "--no-features", "f0"))


def test_rhythm_mode_default_is_auto():
    assert parse().rhythm_mode == "auto"
