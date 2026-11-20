"""Spread re-run guard: it compares like with like after a JSON round trip, and catches real differences."""

import pytest

pytest.importorskip("statsmodels")

from pochettino_fit.spread import mismatches  # noqa: E402


def summary(p1: float) -> dict:
    run = {"position_probabilities": {1: p1, 2: 1 - p1}, "p_top_half": 0.5}
    return {"united_now": {"played": 10}, "baseline": run, "pochettino": run, "contribution": {"mean": 0.5}}


def test_integer_keys_match_frozen_string_keys() -> None:
    frozen = {"united_now": {"played": 10}, "contribution": {"mean": 0.5},
              "baseline": {"position_probabilities": {"1": 0.3, "2": 0.7}, "p_top_half": 0.5}}
    frozen["pochettino"] = frozen["baseline"]
    assert mismatches(summary(0.3), frozen) == []
    expected = ["baseline.position_probabilities", "pochettino.position_probabilities"]
    assert sorted(mismatches(summary(0.31), frozen)) == expected
